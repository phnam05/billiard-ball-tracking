def get_index_of_min(Data_List):

    import numpy as np
    
    #make sure data is in a standard list, not a numpy array
    if (type(Data_List).__module__ == np.__name__):
        Data_List = list(Data_List)
    
    #return a list of the indexes of the minimum values. Important if there is >1 minimum
    return [i for i,x in enumerate(Data_List) if x == min(Data_List)]


def get_index_of_max(Data_List):

    import numpy as np    
    
    #make sure data is in a standard list, not a numpy array
    if (type(Data_List).__module__ == np.__name__):
        Data_List = list(Data_List)
    
    #return a list of the indexes of the max values. Important if there is >1 maximum
    return [i for i,x in enumerate(Data_List) if x == max(Data_List)]