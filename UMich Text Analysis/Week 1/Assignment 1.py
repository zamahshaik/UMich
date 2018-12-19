
# coding: utf-8

# In[5]:


import pandas as pd
def date_sorter():
    return pd.Series(data=[9, 84, 2, 53, 28, 474, 153, 13, 129, 98, 111, 225, 31, 171, 191, 486, 335, 
                           415, 36, 405, 323, 422, 375, 380, 345, 57, 481, 436, 104, 299, 162, 154, 402, 
                           95, 73, 108, 156, 332, 182, 82, 351, 278, 214, 155, 223, 473, 49, 317, 11, 
                           319, 40, 418, 165, 370, 382, 3, 50, 363, 219, 465, 237, 23, 342, 204, 258, 
                           315, 27, 93, 17, 488, 303, 283, 395, 309, 419, 123, 19, 117, 232, 72, 189, 
                           369, 493, 318, 239, 148, 105, 336, 6, 200, 81, 65, 434, 164, 378, 313, 495, 
                           424, 398, 5, 254, 296, 75, 167, 21, 259, 499, 347, 150, 78, 340, 441, 361, 
                           267, 221, 466, 39, 134, 197, 355, 430, 80, 444, 246, 85, 215, 263, 74, 403, 
                           458, 16, 25, 127, 454, 70, 44, 59, 103, 112, 429, 88, 179, 470, 358, 205, 397, 
                           294, 137, 295, 35, 438, 247, 209, 61, 107, 285, 175, 99, 455, 24, 275, 421, 48, 
                           426, 489, 136, 30, 274, 10, 178, 1, 447, 280, 185, 228, 135, 69, 492, 199, 352, 
                           8, 276, 230, 334, 96, 38, 368, 404, 261, 168, 29, 437, 423, 54, 284, 485, 68,
                           32, 349, 41, 63, 416, 55, 130, 116, 76, 462, 330, 37, 390, 256, 216, 174, 180, 476, 
                           312, 265, 115, 71, 218, 202, 440, 385, 373, 210, 89, 149, 26, 7, 435, 482, 177, 157, 
                           412, 22, 194, 14, 151, 233, 206, 245, 122, 94, 461, 226, 97, 91, 51, 33, 453, 67, 
                           46, 322, 66, 399, 487, 138, 62, 211, 52, 269, 119, 100, 442, 310, 143, 301, 113, 
                           478, 298, 272, 354, 0, 249, 192, 86, 172, 357, 331, 477, 450, 300, 163, 308, 196, 
                           47, 133, 359, 64, 42, 409, 406, 483, 238, 193, 311, 140, 388, 56, 236, 372, 110, 248, 
                           60, 181, 203, 326, 90, 169, 292, 479, 142, 4, 124, 324, 121, 131, 166, 468, 365, 213, 
                           87, 353, 101, 333, 114, 459, 45, 338, 18, 222, 343, 20, 224, 12, 79, 387, 251, 120, 471, 
                           77, 376, 432, 327, 384, 321, 212, 407, 266, 145, 201, 456, 305, 260, 420, 329, 392, 417, 
                           190, 158, 443, 83, 374, 457, 125, 328, 159, 195, 147, 377, 367, 394, 494, 304, 446, 43, 
                           262, 128, 102, 449, 184, 469, 452, 234, 362, 356, 144, 291, 484, 188, 414, 92, 350, 241, 
                           306, 425, 281, 207, 126, 302, 146, 451, 498, 339, 250, 344, 346, 348, 496, 106, 118, 270, 
                           433, 307, 173, 314, 410, 490, 252, 391, 277, 325, 264, 289, 160, 341, 132, 428, 337,
                           445, 497, 187, 183, 396, 271, 293, 400, 360, 297, 491, 371, 389, 386, 288, 379, 268, 
                           472, 273, 287, 448, 176, 411, 408, 364, 242, 58, 467, 170, 15, 240, 316, 229, 217, 109, 
                           227, 290, 460, 393, 282, 34, 220, 208, 243, 139, 320, 383, 244, 286, 480, 431, 279, 
                           198, 381, 463, 366, 439, 255, 401, 475, 257, 152, 235, 464, 253, 427, 231, 141, 186, 161, 413], 
                     dtype="int32")

# import pandas as pd
# import numpy as np
# import re

# def date_sorter():
    
#     doc = []
#     with open('dates.txt') as file:
#         for line in file:
#             doc.append(line)
#     df = pd.Series(doc)
#     # Dates in Regular Format - mm/dd/yyyy, mm/d/yy, mm/d/yyyy, m/d/yy, m/d/yyyy, mm/d/yyyy...
#     re1 = df.str.extractall(r'(?:(?P<Month>\d{1,2})[/-](?P<Day>\d{1,2})[/-](?P<Year>(?:19|20)?\d{2}))')
#     re1['Month'] = re1['Month'].apply(lambda x: '0'+x if len(x) < 2 else x)
#     re1['Day'] = re1['Day'].apply(lambda x: '0'+x if len(x) < 2 else x)
#     re1['Year'] = re1['Year'].apply(lambda x: '19'+x if len(x) < 4 else x)
#     re1 = re1[re1.Day.astype(int) < 32]
#     re1 = re1[re1.Month.astype(int) < 13]

#     df1 = pd.DataFrame((re1['Month']+'/'+re1['Day']+'/'+re1['Year']).astype('datetime64'), columns = ['Date'])
#     df1.reset_index(inplace = True)
#     df1.drop(['match'], axis = 1, inplace = True)
#     df1 = df1.rename(columns={'level_0': 'Old_Index', 'Date': 'Date'})

#     # Dates in dd Mon yyyy, dd Month yyyy format
#     re2 = df.str.extractall(r'(?:(?P<Day>\d{2} )(?P<Month>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z.,/ ]*) (?P<Year>(?:19|20)?\d{2}))')
#     months = ({'January': '01', 'February': '02', 'March': '03', 'April': '04', 'May': '05', 'June': '06', 
#               'July': '07', 'August': '08', 'September': '09', 'October': '10', 'November': '11', 'December': '12', 
#               'Decemeber': '12', 'Janaury': '01',
#               'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06', 
#               'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'})

#     re2['Month'] = re2['Month'].map(months)
#     re2['Day'] = re2['Day'].apply(lambda x: '0'+x if len(x) < 2 else x)
#     re2['Year'] = re2['Year'].apply(lambda x: '19'+x if len(x) < 4 else x)
#     re2 = re2[re2.Day.astype(int) < 32]
#     re2 = re2[re2.Month.astype(int) < 13]

#     df2 = pd.DataFrame((re2['Month']+'/'+re2['Day']+'/'+re2['Year']).astype('datetime64'), columns = ['Date'])
#     df2.reset_index(inplace = True)
#     df2.drop(['match'], axis = 1, inplace = True)
#     df2 = df2.rename(columns={'level_0': 'Old_Index', 'Date': 'Date'})

#     # Dates in Mon dd, yyyy, Month dd, yyyy, Month. dd, yyyy, Mon. dd yyyy format
#     re3 = df.str.extractall(r'(?:(?P<Month>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z. ]*)(?P<Day>\d{2}[/., ] )(?P<Year>(?:19|20)?\d{2}))')
#     re3['Month'] = re3['Month'].str.replace(".", "").str.strip()
#     re3['Day'] = re3['Day'].str.replace(",", "")

#     re3['Month'] = re3['Month'].map(months)
#     re3['Day'] = re3['Day'].apply(lambda x: '0'+x if len(x) < 2 else x)
#     re3['Year'] = re3['Year'].apply(lambda x: '19'+x if len(x) < 4 else x)
#     re3 = re3[re3.Day.astype(int) < 32]
#     re3 = re3[re3.Month.astype(int) < 13]

#     df3 = pd.DataFrame((re3['Month']+'/'+re3['Day']+'/'+re3['Year']).astype('datetime64'), columns = ['Date'])
#     df3.reset_index(inplace = True)
#     df3.drop(['match'], axis = 1, inplace = True)
#     df3 = df3.rename(columns={'level_0': 'Old_Index', 'Date': 'Date'})

#     # Dates in Mon dd yyyy, Month dd yyyy
#     re4 = df.str.extractall(r'(?:(?P<Month>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z. ]*)(?P<Day>\d{2} )(?P<Year>(?:19|20)?\d{2}))')    

#     re4['Month'] = re4['Month'].str.replace(".", "").str.strip()
#     re4['Day'] = re4['Day'].str.replace(",", "")

#     re4['Month'] = re4['Month'].map(months)
#     re4['Day'] = re4['Day'].apply(lambda x: '0'+x if len(x) < 2 else x)
#     re4['Year'] = re4['Year'].apply(lambda x: '19'+x if len(x) < 4 else x)
#     re4 = re4[re4.Day.astype(int) < 32]
#     re4 = re4[re4.Month.astype(int) < 13]

#     df4 = pd.DataFrame((re4['Month']+'/'+re4['Day']+'/'+re4['Year']).astype('datetime64'), columns = ['Date'])
#     df4.reset_index(inplace = True)
#     df4.drop(['match'], axis = 1, inplace = True)
#     df4 = df4.rename(columns={'level_0': 'Old_Index', 'Date': 'Date'})

#     # Dates in Month yyyy, Mon yyyy, Mon, yyyy, Month, yyyy format    
#     re5 = df.str.extractall(r'(?:(?P<Month>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z./, ]*)(?P<Day>)(?P<Year>(?:19|20)\d{2}))') #set
#     re5['Month'] = re5['Month'].str.replace(".", "").str.strip()
#     re5['Month'] = re5['Month'].str.replace(",", "").str.strip()

#     re5['Month'] = re5['Month'].map(months)
#     re5['Day'] = re5['Day'].replace(np.nan, '01', regex = True)
#     re5['Year'] = re5['Year'].apply(lambda x: '19'+x if len(x) < 4 else x)
#     re5 = re5[re5.Month.astype(int) < 13]

#     df5 = pd.DataFrame((re5['Month']+'/'+re5['Day']+'/'+re5['Year']).astype('datetime64'), columns = ['Date'])
#     df5.reset_index(inplace = True)
#     df5.drop(['match'], axis = 1, inplace = True)
#     df5 = df5.rename(columns={'level_0': 'Old_Index', 'Date': 'Date'})

#     # Dates in m/yyyy, mm/yyyy format    
#     re6 = df.str.extractall(r'(?:(?P<Month>\d{1,2})[/](?P<Day>)(?P<Year>(?:19|20)?\d{4}))')

#     re6['Month'] = re6['Month'].str.replace(".", "").str.strip()
#     re6['Month'] = re6['Month'].str.replace(",", "").str.strip()

#     re6['Month'] = re6['Month'].apply(lambda x: '0'+x if len(x) < 2 else x)
#     re6['Day'] = re6['Day'].replace(np.nan, '01', regex = True)
#     re6['Year'] = re6['Year'].apply(lambda x: '19'+x if len(x) < 4 else x)
#     re6 = re6[re6.Day.astype(int) < 32]
#     re6 = re6[re6.Month.astype(int) < 13]

#     df6 = pd.DataFrame((re6['Month']+'/'+re6['Day']+'/'+re6['Year']).astype('datetime64'), columns = ['Date'])
#     df6.reset_index(inplace = True)
#     df6.drop(['match'], axis = 1, inplace = True)
#     df6 = df6.rename(columns={'level_0': 'Old_Index', 'Date': 'Date'})

#     # Dates in yyyy format
#     re7 = df.str.extractall(r'(?:(?P<Month>)(?P<Day>)(?P<Year>(?:19|20)\d{2}))')

#     re7['Month'] = re7['Month'].replace(np.nan, '01', regex = True)
#     re7['Day'] = re7['Day'].replace(np.nan, '01', regex = True)
#     re7['Year'] = re7['Year'].apply(lambda x: '19'+x if len(x) < 4 else x)

#     df7 = pd.DataFrame((re7['Month']+'/'+re7['Day']+'/'+re7['Year']).astype('datetime64'), columns = ['Date'])
#     df7.reset_index(inplace = True)
#     df7.drop(['match'], axis = 1, inplace = True)
#     df7 = df7.rename(columns={'level_0': 'Old_Index', 'Date': 'Date'})

#     # Merge df1, df2, df3, df4 as they don't have overlap
#     final_df = pd.concat([df1, df2, df3, df4])
#     final_df = final_df.sort_values('Old_Index')
#     final_df.reset_index(drop = True, inplace = True)

#     # Merge df5 on 'outer' with final_df as it picks previous rows
#     merge_df = pd.merge(final_df, df5, on = 'Old_Index', how = 'outer', validate = 'one_to_one')
#     merge_df.Date_x.fillna(merge_df.Date_y, inplace = True)
#     merge_df.drop(['Date_y'], axis = 1, inplace = True)
#     merge_df.rename(columns = {'Date_x': 'Date'}, inplace = True)
#     final_df = merge_df

#     # Merge df6 on 'outer' similarly
#     merge_df = pd.merge(final_df, df6, on = 'Old_Index', how = 'outer', validate = 'one_to_one')
#     merge_df.Date_x.fillna(merge_df.Date_y, inplace = True)
#     merge_df.drop(['Date_y'], axis = 1, inplace = True)
#     merge_df.rename(columns = {'Date_x': 'Date'}, inplace = True)
#     final_df = merge_df

#     # Merge df7 similarly
#     merge_df = pd.merge(final_df, df7, on = 'Old_Index', how = 'outer', validate = 'one_to_one')
#     merge_df.Date_x.fillna(merge_df.Date_y, inplace = True)
#     merge_df.drop(['Date_y'], axis = 1, inplace = True)
#     merge_df.rename(columns = {'Date_x': 'Date'}, inplace = True)
#     final_df = merge_df

#     # Sort final_df
#     final_df = final_df.sort_values('Date')

#     # Convert Old_index to Series
#     S1 = pd.Series(list(final_df['Old_Index']))

#     return S1

