# Dataset of EEG recordings containing HFO markings for 30 pediatric patients with epilepsy 

## Summary
High-frequency oscillations in scalp EEG are promising non-invasive biomarkers of epileptogenicity. However, it is unclear how high-frequency oscillations are impacted by age in the pediatric population. 
We recorded and processed the first 3 hours of sleep EEG data in 30 children and adolescents with focal or generalized epilepsy. We used an automated and clinically validated high-frequency oscillation detector to determine ripple rates (80-250 Hz) in bipolar channels. The software for the detection of HFOs is freely available at the GitHub repository (https://github.com/ZurichNCH/Automatic-High-Frequency-Oscillation-Detector). Furthermore HFO markings are also added in this database for the selected N3 intervals.


## Repository structure

### Main directory (hfo/)
Contains metadata files in the BIDS standard about the participants and the study. Folders are explained below.

### Subfolders
* hfo/sub-**/
Contains folders for each subject, named sub-<subject number> and session information.
* hfo/sub-**/ses-01/eeg
Contains the raw eeg data in .edf format for each subject. The duration is typically 3 hours, that was recorded in the beginning of the sleep. Details about the channels are given in the corresponding .tsv file. 
* hfo/derivatives
Besides containingsubfolders for the raw data, there are two .json files. The events_description.json explains the meaning of the columns of the event description tsv files (in the subfolders).
The interval_description.json explains the meaning of the columns of the interval description tsv files (in the subfolders).

* hfo/derivatives/sub-**/ses-01/eeg/
Contains processed data for each subject. Based on the sleep annotations, first we identified the sleep stages. Then we cut 5 minutes data intervals from the N3 sleep stages. We applied bipolar referencing by considering all nearest neighbour chanels, thus resulting in 52 bipolar channels. Each run corresponds to one 5 minute data interval. The DataIntervals.tsv file provides information about how the various runs are related to the raw data by providing the start and end indeces. Besides the .edf and channel descriptor .tsv files there is an other .tsv file containing the detected candidate event details. Eg. sub-26_ses-01_task-hfo_run-01_events.tsv contains for subject 26 for the first processed data interval the event markings as indeces with additional features of this event described in the abovementioned events_description.json file.


## Related materials
The code for HFO detection is available at https://github.com/ZurichNCH/Automatic-High-Frequency-Oscillation-Detector


## Support
For questions on the dataset or the task, contact Johannes Sarnthein at [johannes.sarnthein@usz.ch](johannes.sarnthein@usz.ch).
