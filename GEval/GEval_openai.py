# In this file, we will use the OpenAI API to do the evaluation process
# The whole format use the /workspace/MotionExpert_forked/GEval/GEval_score_calculator.py from Coachme project to be the reference.
# Citations:
# CoachMe (MotionXperts/MotionExpert): https://github.com/MotionXperts/MotionExpert
# OpenAI API: https://platform.openai.com/docs/api-reference

import openai, json, argparse, tqdm, os, re
from dotenv import load_dotenv

def read_template(path):
    return open(path).read()

def read_data(ground_truth_path, pred_path):
    ground_truth = json.load(open(ground_truth_path, encoding='utf-8'))
    preds = json.load(open(pred_path, encoding='utf-8'))
    out = []
    for k in ground_truth:
        out.append({"file_name": k, "source": ground_truth[k], "system_output": preds.get(k, "")})
    return out

def g_eval(model, samples, prompt, epoch, out_dir, model_name):
    records = []
    success_count, total = 0, 0
    for inst in tqdm.tqdm(samples, desc=f"epoch {epoch}"):
        src = inst['source']
        sysout = inst['system_output']
        if isinstance(sysout, list):
            sysout = sysout[0] if sysout else ""
        if not sysout:
            continue
        cur = prompt.replace('{{Document}}', src).replace('{{Summary}}', sysout)
        try:
            resp = model.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": cur}],
                max_tokens=5,
                temperature=0,
            )
            text = resp.choices[0].message.content
            m = re.search(r'\d+', text)
            if not m:
                continue
            score = int(m.group())
            inst['score'] = score
            inst['raw'] = text
            records.append(inst)
            total += score
            success_count += 1
        except Exception as e:
            print("Error:", e)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f'geval_openai_epoch_{epoch}.json'), 'w') as f:
        json.dump(records, f, indent=2)
    return total / max(success_count, 1), success_count

if __name__ == '__main__':
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompt', default='./GEval_Consistency_Template.txt')
    ap.add_argument('--ground_truth', default='./tennis_gt.json')
    ap.add_argument('--predict_dir', default='../results/tennis_v2/jsons')
    ap.add_argument('--output', default='../results/tennis_v2/geval')
    ap.add_argument('--epochs', nargs='+', type=int)
    ap.add_argument('--model', default='gpt-4o')
    args = ap.parse_args()

    # Set the API Key
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise SystemExit("You need to set the OPENAI_API_KEY at first.")
    model = openai.OpenAI(api_key=key)
    tpl = read_template(args.prompt)

    
    summary = {}
    for e in args.epochs:
        pred_path = os.path.join(args.predict_dir, f'results_epoch{e}.json')
        if not os.path.exists(pred_path):
            continue
        samples = read_data(args.ground_truth, pred_path)
        avg, success_count = g_eval(model, samples, tpl, e, args.output, args.model)
        summary[f'epoch_{e}'] = {'avg_score': avg, 'scored': success_count, 'model': args.model}
        print(f"Epoch {e}: avg={avg:.3f} ({success_count} scored)")

    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, 'geval_openai_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print("\nFinal:", json.dumps(summary, indent=2))
