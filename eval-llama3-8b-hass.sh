TargetPath=meta-llama/Meta-Llama-3-8B-Instruct
DraftPath=neurips24645/llama3-8b-instruct-hass-reproduce
for iter in 1 2 3
do
	for DATASET in mt_bench alpaca gsm8k humaneval qa sum
	do
		for DEPTH in 5 6 7
		do
			CUDA_VISIBLE_DEVICES=0 python -m evaluation.gen_ea_answer_llama3chat \
				--ea-model-path ${DraftPath} \
				--base-model-path ${TargetPath} \
				--temperature 0 \
				--model-id llama3-8b-hass-base-depth${DEPTH}-tt60-iter${iter} \
				--bench-name ${DATASET} \
				--total-token 60 \
				--depth ${DEPTH}

			echo "Finish detph ${DEPTH}"
		done
		echo "Finish Dataset ${DATASET}"
	done
	echo "Finish Iteration ${iter}"
done

for DEPTH in 5 6 7
do
    for DATASET in mt_bench alpaca gsm8k humaneval qa sum
    do
		for iter in 1 2 3
		do
            FileName="llama3-8b-hass-base-depth${DEPTH}-tt60-iter${iter}-temperature-0.0.jsonl"

            # You must run the base model (non-speculatived decoding) before evaluating the speed,
            # because the speedup ratio is comparing with the base model.
			# SpeedFile="speed.llama3-8b-hass-base.${DATASET}.txt"
            # echo ${FileName} >> ${SpeedFile}
            # python evaluation/speed.py --my_model_json ${DATASET}/${FileName} --bench ${DATASET} | tail -n 1 >> ${SpeedFile}
            # echo "" >> ${SpeedFile}

			ThroughputFile="throughput.llama3-8b-hass-base.${DATASET}.txt"
            echo ${FileName} >> ${ThroughputFile}
            python evaluation/throughput.py --my_model_json ${DATASET}/${FileName} | tail -n 1 >> ${ThroughputFile}
            echo "" >> ${ThroughputFile}

            AcclenFile="acclen.llama3-8b-hass-base.${DATASET}.txt"
            echo ${FileName} >> ${AcclenFile}
            python evaluation/acceptance_length.py --my_model_json ${DATASET}/${FileName} | tail -n 1 >> ${AcclenFile}
            echo "" >> ${AcclenFile}
		done
    done
done 