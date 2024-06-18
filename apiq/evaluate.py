import os
from tqdm import tqdm
import torch
import torch.nn as nn
from apiq.data_utils import get_loaders

@torch.no_grad()
def evaluate(model, tokenizer, args, logging):
    logging.info("=== start evaluation ===")
    results = {}
    if "llama" in args.model_family or "mistral" in args.model_family:
        model = model.to(args.device)
    else:
        raise ValueError("Only support llama/mistral")
    
    if args.eval_ppl:
        for dataset in ["wikitext2", "c4"]:
            cache_testloader = f'{args.cache_dir}/testloader_{args.model_name_or_path.split("/")[-1]}_{dataset}_all.cache'
            if os.path.exists(cache_testloader):
                testloader = torch.load(cache_testloader)
                logging.info(f"load calibration from {cache_testloader}")
            else:
                dataloader, testloader = get_loaders(
                    dataset,
                    tokenizer,
                    seed=args.seed,
                    seqlen=2048,
                )
                torch.save(testloader, cache_testloader)

            if "c4" in dataset:
                testenc = testloader
            else:
                testenc = testloader.input_ids

            nsamples = testenc.numel() // args.seqlen
            use_cache = model.config.use_cache
            model.config.use_cache = False
            model.eval()
            nlls = []
            for i in tqdm(range(nsamples)):
                batch = testenc[:, (i * args.seqlen) : ((i + 1) * args.seqlen)].to(args.device)
                # TODO: check
                if "llama" in args.model_family or "mistral" in args.model_family:
                    outputs = model.base_model.model.model(batch)
                hidden_states = outputs[0]
                logits = model.base_model.model.lm_head(hidden_states)
                shift_logits = logits[:, :-1, :]
                shift_labels = testenc[:, (i * args.seqlen) : ((i + 1) * args.seqlen)][
                    :, 1:
                ].to(model.base_model.model.lm_head.weight.device)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                neg_log_likelihood = loss.float() * args.seqlen
                nlls.append(neg_log_likelihood)
                if i == args.limit:
                    break

            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * args.seqlen))
            logging.info(f'{dataset} : {ppl.item()}')
            model.config.use_cache = use_cache
            results[dataset] = ppl.item()

    return results