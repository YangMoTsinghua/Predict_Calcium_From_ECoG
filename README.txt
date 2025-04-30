Use ECoG signals to predict single neuron calcium activity

This Code is based on a previous work: 
Manley, J., Lu, S., Barber, K., Demas, J., Kim, H., Meyer, D., Martínez Traub, F., & Vaziri, A. (2024). Simultaneous, cortex-wide dynamics of up to 1 million neurons reveal unbounded scaling of dimensionality with neuron number. Neuron. Retrieved from https://www.sciencedirect.com/science/article/abs/pii/S0896627324001211
I would like to thank Jason Manley and Alipasha Vaziri for their outstanding contributions to the field of neuroscience and for making their code available to the public, which allows for broader application and development within the research community.

The original code can be download at: 
https://github.com/vazirilab/scaling_analysis
It includes two main parts: 
1.Scaling Analysis, which enables prediction of calcium activity from behavior PCs.
2.PopulationCoding, which enables dimension reduction of calcium signals by SVCA. 

The SVCA method was first described in this work:
Stringer, C., Pachitariu, M., Steinmetz, N., Bai Reddy, C., Carandini, M., & Harris, K. D. (2019). Spontaneous behaviors drive multidimensional, brainwide activity. Science, 364(6437), 255-258. https://doi.org/10.1126/science.aav7893  

Building on their foundation, I have extended the functionality of the code to use ECoG signals for predicting calcium activity. To run this code, first install the scaling analysis and population coding packages as instructed on GitHub.

Next, copy the "dimred.py" file from the "extra_utils_for_population_coding" folder into the Population Coding installation directory. On my computer, this path is "C:\Users\12770\anaconda3\envs\ym\Lib\site-packages\PopulationCoding", which will overwrite the original file. The main modifications begin at line 186, allowing cov and var to be obtained separately during anesthesia/awake states.

Then, copy "predict.py," "svca.py," and "utils_ym.py" from the "extra_utils_for_scaling_analysis" folder into the Scaling Analysis installation environment. On my computer, this path is "C:\Users\12770\anaconda3\envs\ym\Lib\site-packages\scaling_analysis". The "predict.py" file should overwrite the original; the modifications start at line 229 to obtain separate residuals cov_res_beh (physically meaning cov_res_ECoG) for predicted and actual values during anesthesia/awake states. The "svca.py" file modifies interleave to better split training and test sets. The "utils_ym.py" provides all the major functions that were rewritten to achieve the above prediction and plot functionalities.


Then, To run the main code of this project, please do the following steps:
1) Prepare the clacium data
Export the variable "valid_C" from "all_infered_results_filtered.mat", and store it as a "valid_C.csv" file.

2) Prepare the ecog data
Export the ecog data (fp01 - fp32) during anesthesia from "m010 iso.pl2", and store it as a "ele_signal_burst_supp_fp01-fp32.csv" file. 
The exported duration is required to contain all periods of burst and suppression for the given rat.

3) Move the files "valid_C.csv" and "ele_signal_burst_supp_fp01-fp32.csv" into the "Data/mXXXisoX" folder.

4) Run "Predict_All.ipynb" step by step in jupyter notebook (using anaconda).
