# FatigueSet Data Processing

This repository contains the code used to preprocess the [FatigueSet dataset](https://sensix.tech/datasets/fatigueset/fatigueset.zip) that comes from the study [1], and was produced as a part of the research project "**Early Fadigue Detection using Wearables as a means for the Reduction of Critical Accidents in the Mining Process**", which was developed by a team of our lab during our participation in the AI League of the 2025 [Pan-African Robotics Competition](https://parcrobotics.org/).

The data preprocessing steps tries to follow most of the steps presented in the study [2], but with some modifications according to our needs.

The preprocessed data can be found [here](https://www.kaggle.com/datasets/anaxmenobrito/fatigueset-pre-processed) and has resulted in the [following notebook](https://www.kaggle.com/code/anaxmenobrito/fatiguelstmclassifier-m-o-c-a) in which we used the preprocessed data to create a ***LSTM Fatigue Detection Classifier Meta Learning Model*** using the concepts of "***Continuous Meta-Learning without Tasks***" presented in the study [3].

## How to use

1. Download the fatigueset dataset from it's repository:

    ```bash
    curl https://sensix.tech/datasets/fatigueset/fatigueset.zip
    ```

    1.1. Unzip the dataset:

    ```bash
    unzip fatigueset.zip
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Pre-process the dataset data using the `prepare_data.py` script we created:

    ```bash
    python prepare_data.py --data_path=fatigueset --window_secs=5 --data_save_dir=processed --num_cores=4
    ```

    3.1. Adjust the `--window_secs=5` according to the time window you want to be used to accumulate and calculate statiscal data regarding the values in each time window from start to the end of the recorded data.

    3.2. Adjust the `--num_cores=4` according to the amount of CPU cores you want to dedicate to the data preprocessing task.

4. After running the script following the last intruction, there should be available a folder inside the `processed/` folder containing data of a time window relative to the folder name, for the case above it should be `processed/5s/` (if ran without changes) for example.

5. Don't be afraid to create an issue if you have suggestions to improve the data preprocessing process.


## References
- [1] “FatigueSet: A Multi-modal Dataset for Modeling Mental Fatigue and Fatigability | Request PDF,” ResearchGate, Nov. 2024, doi: 10.1007/978-3-030-99194-4_14.
- [2] C. Kodikara, S. Wijekoon, and L. Meegahapola, “FatigueSense: Multi-Device and Multimodal Wearable Sensing for Detecting Mental Fatigue,” ACM Trans. Comput. Healthcare, vol. 6, no. 2, p. 14:1-14:36, Feb. 2025, doi: 10.1145/3709363.
- [3] J. Harrison, A. Sharma, C. Finn, and M. Pavone, “Continuous Meta-Learning without Tasks,” Oct. 21, 2020, _arXiv_: arXiv:1912.08866. doi: [10.48550/arXiv.1912.08866](https://doi.org/10.48550/arXiv.1912.08866)