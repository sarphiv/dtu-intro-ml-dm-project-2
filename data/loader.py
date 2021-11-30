#%% Import and setput
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

#Load dataset
df = pd.read_excel(r'./data/loanapp.xls',header=None)
data = df.to_numpy()

#Get shapes of data
[obs,attr] = data.shape

#Name of attributes
headers = ["occ", "loanamt", "action", "msa", "suffolk", "appinc", "typur", "unit",
"married", "dep", "emp", "yjob", "self", "atotinc", "cototinc", "hexp", 
"price", "other", "liq", "rep", "gdlin", "lines", "mortg", "cons", 
"pubrec", "hrat", "obrat", "fixadj", "term", "apr", "prop", "inss", 
"inson", "gift", "cosign", "unver", "review", "netw", "unem", "min30", 
"bd", "mi", "old", "vr", "sch", "black", "hispanic", "male", 
"reject", "approve", "mortno", "mortperf", "mortlat1", "mortlat2", "chist", "multi",
"loanprc", "thick", "white"]

# print("Code finished!")

#%% ---- DATACLEANING --- PART 1
# Find outliers and remove if nessesary
# One-hot encode
# Fix missing datapoints

# Cell U1545 contains the value "1  9", this is replaced with "." to match all other missing datapoints
data[1544,20] = 0

# Defining a value for the cleaned data
cleanData = np.copy(data)

# Data about the attributes is given in attributeInformation.xlsx. The data is manually created from analysis boxplots of the data. It includes the following
# - Which attributes to one i k encode
# - Which attributes contains outliers and how to locate them

# Convert extreme outliers to a mean value.
# Convert all missing values to a mean value
for y, head in enumerate(headers):
    tempAttributeData = cleanData[:, y]
    for x, val in enumerate(tempAttributeData):
        newValue = 0
    
        # Replacing outliers with a mean value. The outliers are found manually using boxplot
        if head == "liq" and val == 1000000:
            newValue = np.mean(cleanData[np.where((cleanData[:,y] != ".") | (cleanData[:,y] != 1000000)),y])
            cleanData[x,y] = newValue
        elif head == "gdlin" and val == 666:
            newValue = np.mean(cleanData[np.where((cleanData[:,y] != ".") | (cleanData[:,y] != 666)),y])
            cleanData[x,y] = newValue
        elif head == "lines" and val == 99999.4:
            newValue = np.mean(cleanData[np.where((cleanData[:,y] != ".") | (cleanData[:,y] != 99999.4)),y])
            cleanData[x,y] = newValue
        elif head == "term" and val == 99999.4:
            newValue = np.mean(cleanData[np.where((cleanData[:,y] != ".") | (cleanData[:,y] != 99999.4)),y])
            cleanData[x,y] = newValue
        elif head == "review" and val == 999:
            newValue = np.mean(cleanData[np.where((cleanData[:,y] != ".") | (cleanData[:,y] != 999)),y])
            cleanData[x,y] = newValue
        
        # Replacing all missing values with a median value      
        if val == ".":
            newValue = np.median(cleanData[cleanData[:, y] != ".",y])
            cleanData[x,y] = newValue
        
cleanData = cleanData.astype(float)
for i in range(len(cleanData)):
    cleanData[i] = cleanData[i].astype(float)
    
# Replace outliers that are 1.5 IQR from Q3 and Q1
outliers = 0
for i, obs_feat in enumerate(cleanData.T):
    q1 = np.quantile(obs_feat, 0.25)
    q3 = np.quantile(obs_feat, 0.75)
    iqr = q3 - q1
    iqr_factor = 3/2
    
    #Some data has one big proportion, resulting in iqr = 0,
    # so other values are incorrectly discarded.
    if iqr == 0:
        iqr = 1
        iqr_factor = 2

    arg_outlier = (q1 - iqr_factor * iqr > obs_feat) | (obs_feat > q3 + iqr_factor * iqr)
    outliers += sum(arg_outlier)
    mean_non_outlier = obs_feat[~arg_outlier].mean()

    cleanData[arg_outlier, i] = mean_non_outlier

# print(outliers, "ouliers found and replaced with mean")

kIncodData = np.copy(cleanData)
kIncodHeader = np.copy(headers).tolist()

# one of k incoding
def kIncode(idx, head):
    global kIncodData, kIncodHeader
    unique = np.unique(kIncodData[:,idx]) # Get a list of unique elements in attribute
    
    # Go though all unique elments (except the first)
    for uni in unique[1:]:
        newAttri = [] # Define a new attribute
        # Go though all datapoints in existing attribute
 
        for dataPoint in kIncodData[:,idx]:
            #If equal to the unique element add 1 to the new attribute else 0
            if(dataPoint == uni): newAttri.append(1)
            else: newAttri.append(0)
        # Add the new k incoded attribute to the dataset
        newAttri = np.array([newAttri])
 
        kIncodData = np.concatenate((kIncodData.T, newAttri)).T
        # Add a header for the new k incoded attribute
        kIncodHeader.append(head + "[" + str(int(uni)) + "]")
    
    # Rename the old header for the k incoded attribute
    kIncodHeader[idx] = head + "[" + str(int(unique[0])) + "]" 
    # Go though all datapoints in existing attribut
    for x, dataPoint in enumerate(kIncodData[:,idx]):
        #If equal to first unique element change to 1 else 0
        if(dataPoint == unique[0]): kIncodData[x,idx] = 1
        else: kIncodData[x,idx] = 0

# Set "apr" moutlier to mean value
kIncodData[np.argmax(kIncodData[:,kIncodHeader.index("apr")]),kIncodHeader.index("apr")] = np.mean(kIncodData[:,kIncodHeader.index("apr")])

# All elements that needs to be one of k incoded
elementsToKIncode = ["occ", "action","typur","prop"]#,"male"]

# One of k incode all the above attributes
for e in elementsToKIncode:
    kIncode(kIncodHeader.index(e), e)

# Rename male[0] and male[1] ind kIncodHeader to male and female
# kIncodHeader[kIncodHeader.index("male[0]")] = "female"
# kIncodHeader[kIncodHeader.index("male[1]")] = "male"
kIncodHeader[kIncodHeader.index("male")] = "sex" #1: Male, #0: Female

def doPCA(X, n_pc):
    Xhat = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
    pca = PCA()
    pca.fit(Xhat)

    pc_best = pca.components_[:n_pc]

    return Xhat  @ pc_best.T


def getData(exclude = [], standardize = False, n_pc = None, return_headers = False):
    X = kIncodData
    l = np.empty([obs,1])
    headers = []

    for j in range(len(kIncodHeader)):
        failed = False
        for i in exclude:
            if(j == kIncodHeader.index(i)):
                failed = True
                break
        if(not failed):
            l = np.concatenate((l.T, [X[:,j]])).T
            headers.append(kIncodHeader[j])

    X = l[:, 1:]


    if n_pc is not None: 
        X = doPCA(X, n_pc)

    if standardize: 
        X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
    
    
    if return_headers:
        return X, np.array(headers)
    else:
        return X


def getClassificationFeatures(standardize = False, n_pc = None):
    X = getData(
        exclude=["white", "black", "hispanic"], 
        standardize=standardize, 
        n_pc = n_pc
    )

    return X


def getRegressionFeatures(standardize = False, n_pc = None, return_headers = False):
    return getData(
        exclude=["loanamt", "apr", "price", "loanprc"], 
        standardize=standardize, 
        n_pc = n_pc,
        return_headers = return_headers
    )
    
    # X = getData(exclude=["price","apr"], standardize=standardize, n_pc = n_pc)
    # X = getData(exclude=[], standardize=standardize, n_pc = n_pc)


def getClassificationLabels():
    l = np.empty([obs,1])
    #NOTE: Do not change, logic assumes these hard-coded values
    include = ["white", "black", "hispanic"]
    header_order = []

    for j in range(len(kIncodHeader)):
        failed = True
        for i in include:
            if(j == kIncodHeader.index(i)):
                failed = False
                break
        if(not failed):
            l = np.concatenate((l.T, [kIncodData[:,j]])).T
            #Save header of appended column
            header_order.append(kIncodHeader[j])
            
    
    #Discard first column
    l = l[:, 1:]
    
    #Ensure correct order as required in parameters and ensure correct type
    l = l[:, [header_order.index(h) for h in include]].astype(int)


    return l


def getRegressionLabels(standardize = False):
    loanamt = kIncodData[:, kIncodHeader.index("loanamt")]
    if standardize:
        return (loanamt - loanamt.mean()) / (loanamt.std() + 1e-8)
    else:
        return loanamt
    # price_col = kIncodData[:, kIncodHeader.index("price")]
    # apr_col =   kIncodData[:, kIncodHeader.index("apr")]
    # value_rat = (price_col - apr_col)/apr_col

    # if standardize:
    #     return (value_rat - value_rat.mean()) / value_rat.std()
    # else:
    #     return value_rat


