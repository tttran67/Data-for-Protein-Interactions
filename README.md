# Data-for-Protein-Interactions
This is the data we extract from the database provided by http://mentha.uniroma2.it in the use of discovering protein interactions.
### 01 folder
csv files inside are named in the form of 'year-month-day'. 'Year-month-day.csv' is the raw material. After a set of processing, the 'year-month-day.csv-reault.csv' stores the DCw, SOECC, V_n and V_e. After topsis procedure, 'year-month-day.csv-score.csv' stores the score given by topsis alg.
### 02 folder
AVG.xlsx takes all the proteins of each week into consideration, showing the averange of DCw, SOECC, N_v, N_e and topsis score.

TOP5.xlsx takes only TOP5 proteins of each week into consideration, showing the averange of DCw, SOECC, N_v, N_e and topsis score.

RATIO.xlsx contains the pivot value of given ratio. For example, in the first table 'DCw', col B represents the very first DCw value of the corresponding raw data, sorted by the value of DCw from height to low. col C represents the value that divide the whole data by 5%, also sorted by the value of DCw from height to low. col D, E, F follows the same principle with C. To achieve this, we first sort the whole table by DCw from height to low, then we caculate the 5%\10%\25%\50% position and get the corrresponding value.
### 03 folder
all.zip is the main raw data we use. classified by taxion.rar and classified by taxon-csv.zip are the data classified by taxon.
