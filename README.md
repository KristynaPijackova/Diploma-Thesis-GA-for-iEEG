# Optimizing neural network architecture for EEG processing using evolutionary algorithms

**Author:** Kristyna Pijackova
**2022/2023**
**Brno University of Technology, Faculty of Electrical Engineering and Communications** 
This thesis was done in colaboration with the [Computational Neuroscience group](https://www.isibrno.cz/en/computational-neuroscience) at the Institute of Scientific Instruments in Brno.

**Abstract**

This thesis deals with an optimization of neural network hyperparameters for EEG signal processing using evolutionary algorithms. In this work, a genetic algorithm was proposed that is suitable for hyperparameters optimization as well as neural architecture search. These methods were compared to a benchmark model designed by an engineer with expertise in iEEG processing. Data used in this work are classified into four categories and come from St. Anne's University Hospital (SAUH) and Mayo Clinic (MAYO) and were recorded on drug-resistant epileptic patients undergoing pre-surgical examination. The results of the neural architecture search method were comparable with the benchmark model. The hyperparameter optimization improved the F1 score over the original, empirically designed, model from 0.9076 to 0.9673 for the SAUH data and 0.9222 to 0.9400 for the Mayo Clinic data. The increased scores were mainly due to the increased accuracy of the classification of pathological events and noise, which may have further positive implications in applications of this model in seizure and noise detectors.

**Data**

[Data](https://springernature.figshare.com/collections/Multicenter_intracranial_EEG_dataset_for_classification_of_graphoelements_and_artifactual_signals/4681208) used in this work are publicly available. For more details about the dataset see [Multicenter intracranial EEG dataset for classification of graphoelements and artifactual signals](https://www.nature.com/articles/s41597-020-0532-5).

**Ethical Statement**

This study was carried out in accordance with the approval of the Mayo Clinic Institutional Review Board with written informed consent from all subjects. The protocol was approved by the Mayo Clinic Institutional Review Board and St. Anne’s University Hospital Research Ethics Committee and the Ethics Committee of Masaryk University. All subjects gave written informed consent in accordance with the Declaration of Helsinki. All methods were performed in accordance with the relevant guidelines and regulations.

**Repository Structure**

*GA Hyperparameters*

This folder contains source files written in Python with implemented genetic algorithm that was used for hyperparameter tuning. The hyperparameter tuning was done on a CNN-GRU architecture originaly introduced by [(Nejedly et. al, 2019)](https://www.nature.com/articles/s41598-019-47854-6). This thesis focuses on tuning of data preprocessing parameters as well as hyperparameter of the neural network. The results are stored in json file with hyperparameters and the evaluated scores.

*GA Architecture*

This folder contains source files written in Python with the implemented genetic algorithm used for neural architecture search (NAS) and the implementation of the NAS.




