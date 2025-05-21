DraftPath=neurips24645/llama2-13b-chat-poss2
TargetPath=meta-llama/Llama-2-13b-chat-hf

for iter in 1 2 3
do
	for DATASET in mt_bench alpaca gsm8k humaneval qa sum
	do
		for DEPTH in 5 6 7
		do
            CUDA_VISIBLE_DEVICES=0 python -m evaluation.gen_poss_answer_llama2chat \
				--ea-model-path ${DraftPath} \
				--base-model-path ${TargetPath} \
				--temperature 1 \
				--model-id llama2-13b-poss2-depth${DEPTH}-tt60-iter${iter} \
				--forward_num_total 6 \
				--position_per_layer 2 \
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

            FileName="llama2-13b-poss2-depth${DEPTH}-tt60-iter${iter}-temperature-1.0.jsonl"

            # You must run the base model (non-speculatived decoding) before evaluating the speed,
            # because the speedup ratio is comparing with the base model.
			# SpeedFile="speed.llama2-13b-poss-2.${DATASET}.txt"
            # echo ${FileName} >> ${SpeedFile}
            # python evaluation/speed.py --my_model_json ${DATASET}/${FileName} --bench ${DATASET} | tail -n 1 >> ${SpeedFile}
            # echo "" >> ${SpeedFile}

			ThroughputFile="throughput.llama2-13b-poss-2.${DATASET}.txt"
            echo ${FileName} >> ${ThroughputFile}
            python evaluation/throughput.py --my_model_json ${DATASET}/${FileName} | tail -n 1 >> ${ThroughputFile}
            echo "" >> ${ThroughputFile}

            AcclenFile="acclen.llama2-13b-poss-2.${DATASET}.txt"
            echo ${FileName} >> ${AcclenFile}
            python evaluation/acceptance_length.py --my_model_json ${DATASET}/${FileName} | tail -n 1 >> ${AcclenFile}
            echo "" >> ${AcclenFile}
		done
    done
done
