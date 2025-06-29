# FatigueSet Data Processing

To prepare the data required for training or testing the model follow the next steps:

1. Dowload the fatigueset dataset from it's repository:

    ```bash
    curl https://sensix.tech/datasets/fatigueset/fatigueset.zip
    ```

    1.1. Unzip the dataset:

    ```bash
    unzip fatigueset.zip
    ```

2. Pre-process the dataset data using the `prepare_data.py` script we created:

    ```bash
    python prepare_data.py --data_path=fatigueset --window_secs=5 --data_save_dir=processed --num_cores=4
    ```

    2.1. Adjust the `--window_secs=5` according to the time window you want to be used to accumulate and calculate statiscal data regarding the values in each time window from start to the end of the recorded data.

    2.2. Adjust the `--num_cores=4` according to the amount of CPU cores you want to dedicate to the data preprocessing task.

3. After running the script following the last intruction, there should be available a folder inside the `processed/` folder containing data of a time window relative to the folder name, for the case above it should be `processed/5s/` (if ran without changes) for example.

4. Don't be afraid to create an issue if you have suggestions to improve the data preprocessing process.

5. We have produced this project as a part of the research project "**Early Fadigue Detection using Wearables as a means for the Reduction of Critical Accidents in the Mining Process**", which was developed by a team of our lab during our participation in the AI League of the 2025 [Pan-African Robotics Competition](https://parcrobotics.org/).

6. We've used the data for the creation of a **LSTM Fatigue Detection Classifier Meta Learning Model** using the concepts presented in the study [3], and as resulted in the [following notebook](https://www.kaggle.com/code/anaxmenobrito/fatiguelstmclassifier-m-o-c-a).


## References
- [1] “FatigueSet: A Multi-modal Dataset for Modeling Mental Fatigue and Fatigability | Request PDF,” ResearchGate, Nov. 2024, doi: 10.1007/978-3-030-99194-4_14.
- [2] C. Kodikara, S. Wijekoon, and L. Meegahapola, “FatigueSense: Multi-Device and Multimodal Wearable Sensing for Detecting Mental Fatigue,” ACM Trans. Comput. Healthcare, vol. 6, no. 2, p. 14:1-14:36, Feb. 2025, doi: 10.1145/3709363.
- [3] J. Harrison, A. Sharma, C. Finn, and M. Pavone, “Continuous Meta-Learning without Tasks,” Oct. 21, 2020, _arXiv_: arXiv:1912.08866. doi: [10.48550/arXiv.1912.08866](https://doi.org/10.48550/arXiv.1912.08866)