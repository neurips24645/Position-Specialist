TargetPath=meta-llama/Llama-2-13b-chat-hf
DraftPath=yuhuili/EAGLE-llama2-chat-13B

for iter in 1
do
    for DATASET in mt_bench alpaca gsm8k humaneval qa sum
    do
        for DEPTH in 5 6 7
        do
            CUDA_VISIBLE_DEVICES=0 python -m evaluation.gen_baseline_answer_llama2chat \
                --ea-model-path ${DraftPath} \
                --base-model-path ${TargetPath} \
                --temperature 0 \
                --bench-name ${DATASET} \
                --model-id llama13b-baseline-iter${iter} 
            echo "Finish ${DATASET}"
        done
    done
done

for DEPTH in 5 6 7
do
    for DATASET in mt_bench alpaca gsm8k humaneval qa sum
    do
		for iter in 1
		do
            FileName="llama13b-baseline-iter${iter}-temperature-0.0.jsonl"

            # This is the base model (non-speculatived decoding).
			ThroughputFile="throughput.llama13b-baseline.${DATASET}.txt"
            echo ${FileName} >> ${ThroughputFile}
            python evaluation/throughput.py --my_model_json ${DATASET}/${FileName} | tail -n 1 >> ${ThroughputFile}
            echo "" >> ${ThroughputFile}
		done
    done
done