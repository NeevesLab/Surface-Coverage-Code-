import javabridge
import xml.etree.ElementTree as ET
import bioformats
import re

# ---- Main function that extracts the relevant metadata from a vsi file
def extract_metadata(filepath,cycle_vm=True,meta_number=None):
    if cycle_vm:
        javabridge.start_vm(class_path=bioformats.JARS)
    biof=extract_meta_bioformats(filepath)
    if meta_number is not None:
        filepath=change_file_num(filepath,meta_number)
    metadata=extract_meta_manual(filepath,metadata=biof)
    if cycle_vm:
        javabridge.kill_vm()
    return metadata

def change_file_num(string,meta_number):
    new_string = re.sub('(\d+)(?!.*\d)',str(meta_number),string)
    return new_string

# ---- Function that gets the attainable information using bioformats
def extract_meta_bioformats(filepath, metadata=dict()):
    omexmlstr = bioformats.get_omexml_metadata(filepath)
    o = bioformats.OMEXML(omexmlstr)
    x = o.image().Pixels
    metadata['size_Z'] = x.SizeZ
    metadata['size_T'] = x.SizeT
    metadata['scale'] = x.PhysicalSizeX
    return metadata

# ---- Function that manually reads through oex metadata file and gets other relevant information
#      paths through the xml file to final metadata value

def extract_meta_manual(file_path,tag=default_tag,metadata=dict(),keys=['cycle time','cycle time_unit','relative step width']):
    file_path=file_path.replace('vsi','oex')
    # read in file as xml tree
    tree=ET.parse(file_path)
    root=tree.getroot()
    # convert tree to string and split lines
    xmlstr=ET.tostring(root, encoding='utf8', method='xml').decode()
    split=xmlstr.splitlines()
    # find the line location of ketys
    line_location=[]
    for k in keys:
        count=0
        # here we need to add quotation marks to match the style of the xml file
        modified_key='"{}"'.format(k)
        # if a line contains our key we store the next line number in a list, as 
        # this is where the value is contained
        for line in split:
            if modified_key in line:
                line_location.append(count+1)
                break
            count+=1
     # iterate through the line number values and extract the numbers from them and
     # compile to a list 
    values=[]
    for l in line_location:
        print(split[l])
        numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
        rx = re.compile(numeric_const_pattern, re.VERBOSE)
        digit=float(rx.findall(split[l])[0])
        values.append(digit)
    # store manual_metadata values in a dictionary
    manual_metadata=dict(zip(keys,values))
    # convert cycle time to seconds
    if manual_metadata['cycle time_unit']==1: # unit of 1 corresponds to hours
        manual_metadata['cycle time']=manual_metadata['cycle time']/3600
    elif manual_metadata['cycle time_unit']==2:# unit of 2 corresponds to minutes
        manual_metadata['cycle time']=manual_metadata['cycle time']/60
    elif manual_metadata['cycle time_unit']==4:# unit of 4 corresponds to milliseconds
        manual_metadata['cycle time']=manual_metadata['cycle time']/60
    # and we skip the unit of 3 as it corresponds to seconds
    manual_metadata.pop('cycle time_unit')
    metadata=metadata|manual_metadata
    return metadata
