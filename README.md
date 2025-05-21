# This is the codebase of NeurIPS 2025 anonymous submission.

## To reproduce the methods described in our paper, please follow the steps below:

### 1. Install dependencies.

Begin by installing the required packages with the following command:

``pip install -r requirements.txt``

### 2. Select a script to run.

We provide scripts covering different methods. You are welcome to select any script based on your specific needs.

### 3. Adjust configurations as needed.

If you do not wish to evaluate all combinations of *DATASET* and *DEPTH*, feel free to edit the selected script to retain only the settings relevant to your need.

### 4. Run the script.

Execute the evaluation script using:

``bash eval-xxx.sh``

For example: 

``bash eval-llama3-8b-poss2.sh``

The script will first eval the model for 3 times, and then calculate the throuphput and acceptance length.

### 5. Analyze the results.

You may directly compare the throughput between methods, or compute the speedup ratio by dividing the throughput of your selected model by that of the baseline model.