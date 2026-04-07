import os, torch, numpy as np, logging, warnings, json, random
os.environ['NUMEXPR_MAX_THREADS'] = '2'
os.environ['TOKENIZERS_PARALLELISM'] = "false"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from utils.parser import parse_args, load_config
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import torch.distributed as dist
from dataloaders import construct_dataloader
from models.CoachMe import CoachMe
from models import save_checkpoint, load_checkpoint
from datetime import timedelta

logger = logging.getLogger(__name__)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train(cfg, train_dataloader, model, optimizer, scheduler, summary_writer, epoch):
    model.train()
    loss_list = []
    Tokenizer = AutoTokenizer.from_pretrained('t5-base', use_fast=True)
    if dist.get_rank() == 0:
        train_dataloader = tqdm(train_dataloader, total=len(train_dataloader), desc=f'Epoch {epoch}')

    for index, batch in enumerate(train_dataloader):
        (video_name, skeleton_coords, seq_len, frame_mask, label_batch,
         labels_batch, std_coords, subtraction) = batch

        model.zero_grad()
        optimizer.zero_grad()

        if cfg.LOSS == "RandomGT":
            num_gt = len(labels_batch[0])
            rand_idx = random.randint(0, num_gt - 1)
            instructions = [labels_batch[i][rand_idx] for i in range(len(labels_batch))]
        else:
            instructions = label_batch

        tgt_batch = Tokenizer(instructions, return_tensors="pt", padding="max_length",
                              truncation=True, max_length=160)['input_ids'].to(skeleton_coords.device)
        tgt_input = tgt_batch[:, :-1]
        tgt_label = tgt_batch[:, 1:]

        inputs = {
            "video_name": video_name,
            "skeleton_coords": skeleton_coords.to(model.device),
            "frame_mask": frame_mask.to(model.device),
            "seq_len": seq_len,
            "std_coords": std_coords.to(model.device),
            "decoder_input_ids": tgt_input.to(model.device),
            "labels": tgt_label.to(model.device),
            "subtraction": subtraction.to(model.device),
            "tokenizer": Tokenizer,
        }

        outputs = model(**inputs)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        scheduler.step()

        loss[torch.isnan(loss)] = 0
        dist.all_reduce(loss, async_op=False)
        reduced_loss = loss / dist.get_world_size()
        loss_list.append(reduced_loss.detach().cpu().item())

        if dist.get_rank() == 0:
            train_dataloader.set_postfix({
                'loss': f"{np.mean(loss_list):.4f}",
                'lr': f"{scheduler.optimizer.param_groups[0]['lr']:.2e}"
            })

    avg_loss = np.mean(loss_list)
    if dist.get_rank() == 0:
        summary_writer.add_scalar('train/loss', avg_loss, epoch)
        logger.info(f"Epoch {epoch} : Loss {avg_loss:.4f}")
    return avg_loss


def evaluate(cfg, eval_dataloader, model, epoch, summary_writer, store, video_name_list):
    """Lightweight evaluation: compute loss + generate text, skip heavy NLG metrics."""
    Tokenizer = AutoTokenizer.from_pretrained('t5-base', use_fast=True)
    model.eval()
    loss_list = []
    prompt = "Motion Instruction : "

    with torch.no_grad():
        if dist.get_rank() == 0:
            eval_dataloader = tqdm(eval_dataloader, total=len(eval_dataloader), desc='Evaluating')

        for index, batch in enumerate(eval_dataloader):
            (video_name, skeleton_coords, seq_len, frame_mask, label_batch,
             labels_batch, std_coords, subtraction) = batch

            # Generate text
            decoder_input_ids = Tokenizer(
                [prompt], return_tensors="pt", padding=True, truncation=True,
                max_length=160, add_special_tokens=False
            )['input_ids']
            decoder_input_ids = decoder_input_ids.repeat(skeleton_coords.shape[0], 1).to(skeleton_coords.device)

            inputs = {
                "video_name": video_name,
                "skeleton_coords": skeleton_coords.to(model.device),
                "frame_mask": frame_mask.to(model.device),
                "seq_len": seq_len,
                "std_coords": std_coords.to(model.device),
                "decoder_input_ids": decoder_input_ids.to(model.device),
                "subtraction": subtraction.to(model.device),
                "tokenizer": Tokenizer,
                "result_dir": cfg.LOGDIR,
                "epoch": epoch,
            }

            generated_ids, att_node, att_graph, max_indices = model.module.generate(**inputs)

            # Compute loss
            if cfg.EVAL.score and label_batch[0] is not None:
                tgt_batch = Tokenizer(label_batch, return_tensors="pt", padding="max_length",
                                      truncation=True, max_length=160)['input_ids'].to(skeleton_coords.device)
                tgt_input = tgt_batch[:, :-1]
                tgt_label = tgt_batch[:, 1:]
                inputs['decoder_input_ids'] = tgt_input.to(model.device)
                inputs['labels'] = tgt_label.to(model.device)
                loss = model(**inputs).loss
                loss[torch.isnan(loss)] = 0
                dist.all_reduce(loss, async_op=False)
                reduced_loss = loss / dist.get_world_size()
                loss_list.append(reduced_loss.detach().cpu().item())

            for name, gen_id in zip(video_name, generated_ids):
                if isinstance(gen_id, torch.Tensor):
                    gen_id = gen_id.tolist()
                decoded_text = Tokenizer.decode(gen_id, skip_special_tokens=True,
                                                clean_up_tokenization_spaces=True).split(prompt)
                decoded_text = decoded_text[1].strip() if len(decoded_text) > 1 else ""
                store.set(name, decoded_text)

            if dist.get_rank() == 0 and loss_list:
                eval_dataloader.set_postfix({'loss': f"{np.mean(loss_list):.4f}"})

    if dist.get_rank() == 0:
        avg_loss = np.mean(loss_list) if loss_list else 0
        summary_writer.add_scalar('eval/loss', avg_loss, epoch)
        logger.info(f"Eval Epoch {epoch} : Loss {avg_loss:.4f}")

        # Save generated results
        results = {}
        for name in video_name_list:
            try:
                results[name] = store.get(name).decode('utf-8').replace('\u2019', "'")
            except:
                continue

        os.makedirs(cfg.JSONDIR, exist_ok=True)
        result_json = os.path.join(cfg.JSONDIR, f'results_epoch{epoch}.json')
        with open(result_json, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results saved in {result_json}")

        # Print sample predictions
        print(f"\n--- Sample predictions (epoch {epoch}) ---")
        for i, (name, text) in enumerate(results.items()):
            if i >= 3:
                break
            print(f"  {name}: {text[:120]}...")
        print()


def main():
    args = parse_args()
    cfg = load_config(args)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(filename)s %(lineno)d: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=os.path.join(cfg.LOGDIR, 'stdout.log')
    )

    # Also log to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(console_handler)

    model = CoachMe(cfg).to(torch.float32)

    # Load video names for evaluation
    import pickle
    with open(cfg.DATA.TEST, 'rb') as f:
        test_data = pickle.load(f)
    video_name_list = [item['video_name'] for item in test_data]

    # DDP init
    dist.init_process_group(backend='nccl', init_method='env://')
    if dist.get_rank() == 0:
        store = dist.TCPStore("127.0.0.1", 8082, dist.get_world_size(), True, timedelta(seconds=30))
    else:
        store = dist.TCPStore("127.0.0.1", 8082, dist.get_world_size(), False, timedelta(seconds=30))

    set_seed(42)

    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    model = model.to(device)
    torch.cuda.set_device(rank)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[device], output_device=device, find_unused_parameters=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.OPTIMIZER.LR))
    summary_writer = SummaryWriter(os.path.join(cfg.LOGDIR, 'train_logs'))

    train_dataloader = construct_dataloader('train', cfg, cfg.DATA.TRAIN)
    test_dataloader = construct_dataloader('test', cfg, cfg.DATA.TEST)

    max_epoch = cfg.OPTIMIZER.MAX_EPOCH
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.OPTIMIZER.WARMUP_STEPS,
        num_training_steps=max_epoch * len(train_dataloader)
    )

    model = model.to(torch.float32)
    start_epoch = load_checkpoint(cfg, model, optimizer)
    model = model.to(torch.float32)

    eval_cycle = getattr(cfg, 'EVAL_CYCLE', 5)

    if dist.get_rank() == 0:
        print(f"\n{'='*60}")
        print(f"Training Config:")
        print(f"  Sport: {cfg.TASK.SPORT}")
        print(f"  Setting: {cfg.SETTING}")
        print(f"  Loss: {cfg.LOSS}")
        print(f"  LR: {cfg.OPTIMIZER.LR}")
        print(f"  Epochs: {start_epoch} -> {max_epoch}")
        print(f"  Batch size: {cfg.DATA.BATCH_SIZE}")
        print(f"  Train samples: {len(train_dataloader.dataset)}")
        print(f"  Test samples: {len(test_dataloader.dataset)}")
        print(f"  Eval every {eval_cycle} epochs")
        print(f"{'='*60}\n")

    try:
        for epoch in range(start_epoch, max_epoch):
            train_dataloader.sampler.set_epoch(epoch)
            avg_loss = train(cfg, train_dataloader, model, optimizer, scheduler, summary_writer, epoch)

            if (epoch + 1) % eval_cycle == 0:
                if dist.get_rank() == 0:
                    os.makedirs(cfg.CKPTDIR, exist_ok=True)
                    save_checkpoint(cfg, model, optimizer, epoch + 1)

                try:
                    evaluate(cfg, test_dataloader, model, epoch + 1, summary_writer, store, video_name_list)
                except Exception as e:
                    import traceback
                    print(traceback.format_exc())
                    print(f"Eval error at epoch {epoch}: {e}, continuing training.")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        print(f"{e} occurred, saving model before quitting.")
    finally:
        if dist.get_rank() == 0 and 'epoch' in dir() and epoch != start_epoch:
            save_checkpoint(cfg, model, optimizer, epoch + 1)
        summary_writer.close()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
