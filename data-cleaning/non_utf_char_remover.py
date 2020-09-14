import os
"""
2017-06-02: Raghava
This code will find all non-utf8 encoding characters.
It will read each  line  and character by character to see if the char is and prints the char position
"""
source_filepath = '/Users/raghava/data-analytics Dropbox/Raghava Rao Mukkamala/cbs-research/student-supervision/Master-students/progress/Yousef-Adrin/text-classification-02/training-data/BinaryModels_model4_cleaned.csv'

source_filepath = '/Users/raghava/data-analytics Dropbox/Raghava Rao Mukkamala/' \
                  'cbs-research/misc+archieve/new/volkwagen.csv'

def getFileExtension(filepath):
    """
    This function returns the filename without the extension.
    """
    return os.path.splitext(os.path.basename(filepath))[1]



def getFilenameWithoutExtension(filepath):
    """
    This function returns the filename without the extension.
    """
    return os.path.splitext(os.path.basename(filepath))[0]



# get the folder path
dirPath = os.path.dirname(source_filepath)

filename_no_extension = getFilenameWithoutExtension(source_filepath)

fileExtension = getFileExtension(source_filepath)

target_filepath = dirPath + '/' + filename_no_extension + '_cleaned' + fileExtension

non_utf_8_char_count = 0

with open (source_filepath,"r",encoding="utf-8") as fileReader, \
        open(target_filepath, 'w', encoding="utf-8") as fileWriter:
  while True:
    c = fileReader.read(1)
    if not c:
      break
    elif (ord(c) > 256):
      non_utf_8_char_count = non_utf_8_char_count + 1
      print('invalid char: ' + c)
    else:
      fileWriter.write(c)

fileReader.close()
fileWriter.close()
print('Done cleaning the file! Non Utf-8 characters found: ' + str(non_utf_8_char_count))





# lines = []
# lineindex = 0
# charindex = 0

# with open (source_filepath,"r",encoding="utf-8") as f:
#    lines = f.readlines()

# for line in lines:
#    lineindex = lineindex + 1
#    charindex = 0
#    for c in line:
#       charindex = charindex + 1
#       if(ord(c) > 256):
#          print('invalid char at ln: ' + str(lineindex) + ' ch index: ' + str(charindex) + ',  ord value:' + str(ord(c)) + ' character: ' + c)
