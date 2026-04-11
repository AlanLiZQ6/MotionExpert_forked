"""
Diagnose mode collapse: check where the information bottleneck is.

Loads a trained checkpoint, runs test set, measures token similarity at each stage:
  1. HPP output (motion_tokens)
  2. Difference tokens (user - expert)
  3. Projection output (tokens fed to T5)

Usage:
  python diagnose_collapse.py --cfg_file ./results/tennis_aligned/tennis_aligned.yaml
"""

import os, sys, torch, numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from utils.parser import parse_args, load_config
from models.CoachMe import CoachMe
from dataloaders.Dataset import DatasetLoader
from dataloaders import collate_fn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch.nn.functional as F
from natsort import natsorted


def cosine_sim_stats(embeddings, name):
    """Compute pairwise cosine similarity stats."""
    normed = F.normalize(embeddings, dim=1)
    sim = torch.mm(normed, normed.t())
    mask = torch.triu(torch.ones_like(sim, dtype=torch.bool), diagonal=1)
    sims = sim[mask]

    dists = torch.cdist(embeddings, embeddings)
    dists_upper = dists[mask]

    print(f"\n--- {name} ---")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Cosine sim: mean={sims.mean():.4f}, std={sims.std():.4f}, min={sims.min():.4f}, max={sims.max():.4f}")
    print(f"  % pairs sim > 0.95: {(sims > 0.95).float().mean()*100:.1f}%")
    print(f"  % pairs sim > 0.99: {(sims > 0.99).float().mean()*100:.1f}%")
    print(f"  L2 dist: mean={dists_upper.mean():.4f}, std={dists_upper.std():.4f}")
    print(f"  Embedding norm: mean={embeddings.norm(dim=1).mean():.4f}, std={embeddings.norm(dim=1).std():.4f}")
    return sims.mean().item()


def main():
    args = parse_args()
    cfg = load_config(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model (no DDP)
    model = CoachMe(cfg).to(device).to(torch.float32)

    # Load pretrain + finetune weights
    pretrain_ckpt = torch.load(cfg.WEIGHT_PATH, weights_only=False, map_location=device)
    model.load_state_dict(pretrain_ckpt["model_state"], strict=False)

    ckpt_dir = os.path.join(cfg.LOGDIR, "checkpoints")
    ckpts = natsorted(os.listdir(ckpt_dir))
    ckpt_path = os.path.join(ckpt_dir, ckpts[-1])
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    print(f"Loaded epoch {ckpt['epoch']}")

    model.eval()

    # Build dataset without DDP
    dataset = DatasetLoader(cfg, cfg.TASK.PRETRAIN, cfg.DATA.TEST, train=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    Tokenizer = AutoTokenizer.from_pretrained('t5-base', use_fast=True)
    prompt = "Motion Instruction : "

    all_motion = []
    all_diff = []
    all_proj = []
    all_texts = []
    all_names = []

    print(f"\nProcessing {len(dataset)} test samples...")

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i % 50 == 0:
                print(f"  Sample {i}/{len(dataset)}")

            (video_name, skeleton_coords, seq_len, frame_mask,
             label_batch, labels_batch, std_coords, subtraction) = batch

            skeleton_coords = skeleton_coords.to(device)
            std_coords = std_coords.to(device)
            frame_mask = frame_mask.to(device)
            subtraction = subtraction.to(device)

            # Stage 1: HPP
            model.HumanPosePerception.eval()
            motion_tokens, _, _ = model.HumanPosePerception(skeleton_coords)
            motion_tokens = motion_tokens.float()

            # Stage 2: Diff
            if model.ref:
                standard_tokens, _, _ = model.HumanPosePerception(std_coords)
                standard_tokens = standard_tokens.float()
                diff_tokens = model.get_diff_feat(motion_tokens, standard_tokens, model.diff_way)
                diff_tokens = diff_tokens.float()
            else:
                diff_tokens = torch.zeros_like(motion_tokens)

            # Stage 3: Projection
            if model.ref:
                proj_tokens, _ = model.get_proj_feat(motion_tokens, diff_tokens, model.ref, model.diff_type)
            else:
                proj_tokens, _ = model.get_proj_feat(motion_tokens, None, model.ref, model.diff_type)
            proj_tokens = proj_tokens.float()

            # Flatten: mean pool over time and joint dims to get fixed-size vector
            # motion_tokens shape: [B, T, V, C] -> mean over T,V -> [B, C]
            motion_flat = motion_tokens.mean(dim=(1, 2))  # [1, C]
            diff_flat = diff_tokens.mean(dim=(1, 2))      # [1, C]
            proj_flat = proj_tokens.mean(dim=1)            # [1, 768]

            all_motion.append(motion_flat.cpu())
            all_diff.append(diff_flat.cpu())
            all_proj.append(proj_flat.cpu())
            all_names.append(video_name[0])

            # Generate text for first 30
            if i < 30:
                decoder_input_ids = Tokenizer(
                    [prompt], return_tensors="pt", padding=True, truncation=True,
                    max_length=160, add_special_tokens=False
                )['input_ids'].to(device)

                generated_ids = model.LanguageModel.generate(
                    inputs_embeds=proj_tokens,
                    attention_mask=frame_mask,
                    decoder_input_ids=decoder_input_ids,
                    max_length=160, num_beams=3,
                    repetition_penalty=5.0, length_penalty=3.0,
                    early_stopping=True
                )
                text = Tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                text = text.split(prompt)[-1].strip() if prompt in text else text
                all_texts.append(text)

    # Stack and analyze
    motion_embs = torch.cat(all_motion, dim=0)
    diff_embs = torch.cat(all_diff, dim=0)
    proj_embs = torch.cat(all_proj, dim=0)

    print(f"\n{'='*60}")
    print(f"DIAGNOSIS REPORT (epoch {ckpt['epoch']}, {len(dataset)} samples)")
    print(f"{'='*60}")

    motion_sim = cosine_sim_stats(motion_embs, "Stage 1: HPP motion_tokens")
    diff_sim = cosine_sim_stats(diff_embs, "Stage 2: Diff tokens (user - expert)")
    proj_sim = cosine_sim_stats(proj_embs, "Stage 3: Projection output (T5 input)")

    # Diff/Motion norm ratio
    print(f"\n--- Signal strength ---")
    m_norm = motion_embs.norm(dim=1).mean()
    d_norm = diff_embs.norm(dim=1).mean()
    print(f"  Motion norm: {m_norm:.4f}")
    print(f"  Diff norm: {d_norm:.4f}")
    print(f"  Diff/Motion ratio: {d_norm/m_norm:.4f}")

    # Generated text diversity
    if all_texts:
        unique = set(all_texts)
        print(f"\n--- Generated text (first 30) ---")
        print(f"  Unique: {len(unique)} / {len(all_texts)}")
        for j, text in enumerate(all_texts[:5]):
            print(f"  [{j}] {all_names[j]}: {text[:120]}...")

    # Interpretation
    print(f"\n{'='*60}")
    print("INTERPRETATION:")
    if proj_sim > 0.95:
        print("  >> T5 inputs nearly identical -> bottleneck is BEFORE T5")
        if diff_sim > 0.95:
            print("  >> Diff tokens collapsed -> subtraction produces no useful signal")
            if motion_sim > 0.95:
                print("  >> HPP outputs also collapsed -> HPP is the root cause")
            else:
                print("  >> HPP is diverse but diff kills it -> diff mechanism is broken")
        else:
            print("  >> Diff is diverse but Projection collapses -> Projection is the bottleneck")
    elif proj_sim < 0.8:
        print("  >> T5 receives diverse inputs but collapses -> T5 fine-tuning is the problem")
    else:
        print("  >> Moderate similarity -> multiple factors may contribute")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
