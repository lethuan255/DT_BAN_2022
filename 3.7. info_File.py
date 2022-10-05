import pefile
import os
import array
import math
import pickle
import joblib
import sys
import argparse
import hashlib
import os
import hashlib
import pefile
import peutils
import json


def get_entropy(data):
    if len(data) == 0:
        return 0.0
    occurences = array.array('L', [0]*256)
    for x in data:
        occurences[x if isinstance(x, int) else ord(x)] += 1
    entropy = 0
    for x in occurences:
        if x:
            p_x = float(x) / len(data)
            entropy -= p_x*math.log(p_x, 2)
    return entropy


def get_resources(pe):
    """Extract resources :
    [entropy, size]"""
    resources = []
    if hasattr(pe, 'DIRECTORY_ENTRY_RESOURCE'):
        try:
            for resource_type in pe.DIRECTORY_ENTRY_RESOURCE.entries:
                if hasattr(resource_type, 'directory'):
                    for resource_id in resource_type.directory.entries:
                        if hasattr(resource_id, 'directory'):
                            for resource_lang in resource_id.directory.entries:
                                data = pe.get_data(resource_lang.data.struct.OffsetToData, resource_lang.data.struct.Size)
                                size = resource_lang.data.struct.Size
                                entropy = get_entropy(data)

                                resources.append([entropy, size])
        except Exception as e:
            return resources
    return resources

def get_version_info(pe):
    """Return version infos"""
    res = {}
    for fileinfo in pe.FileInfo:
        if fileinfo.Key == 'StringFileInfo':
            for st in fileinfo.StringTable:
                for entry in st.entries.items():
                    res[entry[0]] = entry[1]
        if fileinfo.Key == 'VarFileInfo':
            for var in fileinfo.Var:
                res[var.entry.items()[0][0]] = var.entry.items()[0][1]
    if hasattr(pe, 'VS_FIXEDFILEINFO'):
        res['flags'] = pe.VS_FIXEDFILEINFO.FileFlags
        res['os'] = pe.VS_FIXEDFILEINFO.FileOS
        res['type'] = pe.VS_FIXEDFILEINFO.FileType
        res['file_version'] = pe.VS_FIXEDFILEINFO.FileVersionLS
        res['product_version'] = pe.VS_FIXEDFILEINFO.ProductVersionLS
        res['signature'] = pe.VS_FIXEDFILEINFO.Signature
        res['struct_version'] = pe.VS_FIXEDFILEINFO.StrucVersion
    return res

#extract the info for a given file
def extract_infos(fpath):
    res = {}
    pe = pefile.PE(fpath)
    res['Machine'] = pe.FILE_HEADER.Machine
    res['SizeOfOptionalHeader'] = pe.FILE_HEADER.SizeOfOptionalHeader
    res['Characteristics'] = pe.FILE_HEADER.Characteristics
    res['MajorLinkerVersion'] = pe.OPTIONAL_HEADER.MajorLinkerVersion
    res['MinorLinkerVersion'] = pe.OPTIONAL_HEADER.MinorLinkerVersion
    res['SizeOfCode'] = pe.OPTIONAL_HEADER.SizeOfCode
    res['SizeOfInitializedData'] = pe.OPTIONAL_HEADER.SizeOfInitializedData
    res['SizeOfUninitializedData'] = pe.OPTIONAL_HEADER.SizeOfUninitializedData
    res['AddressOfEntryPoint'] = pe.OPTIONAL_HEADER.AddressOfEntryPoint
    res['BaseOfCode'] = pe.OPTIONAL_HEADER.BaseOfCode
    try:
        res['BaseOfData'] = pe.OPTIONAL_HEADER.BaseOfData
    except AttributeError:
        res['BaseOfData'] = 0
    res['ImageBase'] = pe.OPTIONAL_HEADER.ImageBase
    res['SectionAlignment'] = pe.OPTIONAL_HEADER.SectionAlignment
    res['FileAlignment'] = pe.OPTIONAL_HEADER.FileAlignment
    res['MajorOperatingSystemVersion'] = pe.OPTIONAL_HEADER.MajorOperatingSystemVersion
    res['MinorOperatingSystemVersion'] = pe.OPTIONAL_HEADER.MinorOperatingSystemVersion
    res['MajorImageVersion'] = pe.OPTIONAL_HEADER.MajorImageVersion
    res['MinorImageVersion'] = pe.OPTIONAL_HEADER.MinorImageVersion
    res['MajorSubsystemVersion'] = pe.OPTIONAL_HEADER.MajorSubsystemVersion
    res['MinorSubsystemVersion'] = pe.OPTIONAL_HEADER.MinorSubsystemVersion
    res['SizeOfImage'] = pe.OPTIONAL_HEADER.SizeOfImage
    res['SizeOfHeaders'] = pe.OPTIONAL_HEADER.SizeOfHeaders
    res['CheckSum'] = pe.OPTIONAL_HEADER.CheckSum
    res['Subsystem'] = pe.OPTIONAL_HEADER.Subsystem
    res['DllCharacteristics'] = pe.OPTIONAL_HEADER.DllCharacteristics
    res['SizeOfStackReserve'] = pe.OPTIONAL_HEADER.SizeOfStackReserve
    res['SizeOfStackCommit'] = pe.OPTIONAL_HEADER.SizeOfStackCommit
    res['SizeOfHeapReserve'] = pe.OPTIONAL_HEADER.SizeOfHeapReserve
    res['SizeOfHeapCommit'] = pe.OPTIONAL_HEADER.SizeOfHeapCommit
    res['LoaderFlags'] = pe.OPTIONAL_HEADER.LoaderFlags
    res['NumberOfRvaAndSizes'] = pe.OPTIONAL_HEADER.NumberOfRvaAndSizes

    # Sections
    res['SectionsNb'] = len(pe.sections)
    entropy = list(map(lambda x:x.get_entropy(), pe.sections))
    res['SectionsMeanEntropy'] = sum(entropy)/float(len(entropy))
    res['SectionsMinEntropy'] = min(entropy)
    res['SectionsMaxEntropy'] = max(entropy)
    raw_sizes = list(map(lambda x:x.SizeOfRawData, pe.sections))
    res['SectionsMeanRawsize'] = sum(raw_sizes)/float(len(raw_sizes))
    res['SectionsMinRawsize'] = min(raw_sizes)
    res['SectionsMaxRawsize'] = max(raw_sizes)   
    virtual_sizes = list(map(lambda x:x.Misc_VirtualSize, pe.sections))
    res['SectionsMeanVirtualsize'] = sum(virtual_sizes)/float(len(virtual_sizes))
    res['SectionsMinVirtualsize'] = min(virtual_sizes)
    res['SectionMaxVirtualsize'] = max(virtual_sizes)

    #Imports
    try:
        res['ImportsNbDLL'] = len(pe.DIRECTORY_ENTRY_IMPORT)
        imports = sum([x.imports for x in pe.DIRECTORY_ENTRY_IMPORT], [])
        res['ImportsNb'] = len(imports)
        res['ImportsNbOrdinal'] = len(list(filter(lambda x:x.name is None, imports)))
    except AttributeError:
        res['ImportsNbDLL'] = 0
        res['ImportsNb'] = 0
        res['ImportsNbOrdinal'] = 0

    #Exports
    try:
        res['ExportNb'] = len(pe.DIRECTORY_ENTRY_EXPORT.symbols)
    except AttributeError:
        # No export
        res['ExportNb'] = 0
    #Resources
    resources= get_resources(pe)
    res['ResourcesNb'] = len(resources)
    if len(resources)> 0:
        entropy = list(map(lambda x:x[0], resources))
        res['ResourcesMeanEntropy'] = sum(entropy)/float(len(entropy))
        res['ResourcesMinEntropy'] = min(entropy)
        res['ResourcesMaxEntropy'] = max(entropy)  
        sizes = list(map(lambda x:x[1], resources))
        res['ResourcesMeanSize'] = sum(sizes)/float(len(sizes))
        res['ResourcesMinSize'] = min(sizes)
        res['ResourcesMaxSize'] = max(sizes)
    else:
        res['ResourcesNb'] = 0
        res['ResourcesMeanEntropy'] = 0
        res['ResourcesMinEntropy'] = 0
        res['ResourcesMaxEntropy'] = 0
        res['ResourcesMeanSize'] = 0
        res['ResourcesMinSize'] = 0
        res['ResourcesMaxSize'] = 0

    # Load configuration size
    try:
        res['LoadConfigurationSize'] = pe.DIRECTORY_ENTRY_LOAD_CONFIG.struct.Size
    except AttributeError:
        res['LoadConfigurationSize'] = 0


    # Version configuration size
    try:
        version_infos = get_version_info(pe)
        res['VersionInformationSize'] = len(version_infos.keys())
    except AttributeError:
        res['VersionInformationSize'] = 0
    return res

ListDacTrung = [
    "FileName", 
    "MD5",
    "SHA-1",
    "SHA-256",
    "SHA-512",
    "e_magic",
    "e_cblp",
    "e_cp",
    "e_crlc",
    "e_cparhdr",
    "e_minalloc",
    "e_maxalloc",
    "e_ss",
    "e_sp",
    "c_csum",
    "e_ip",
    "c_cs",
    "e_lfarlc",
    "e_ovno",
    "e_oemid",
    "e_oeminfo",
    "e_lfanew",
    "Machine",
    "SizeOfOptionalHeader",
    "Characteristics",
    "Signature",
    "Magic",
    "MajorLinkerVersion",
    "MinorLinkerVersion",
    "SizeOfCode",
    "SizeOfInitializedData",
    "SizeOfUninitializedData",
    "AddressOfEntryPoint",
    "BaseOfCode",
    "BaseOfData",
    "ImageBase",
    "SectionAlignment",
    "FileAlignment",
    "MajorOperatingSystemVersion",
    "MinorOperatingSystemVersion",
    "MajorImageVersion",
    "MinorImageVersion",
    "MajorSubsystemVersion",
    "MinorSubsystemVersion",
    "Reserved1",
    "SizeOfImage",
    "SizeOfHeaders",
    "CheckSum",
    "Subsystem",
    "DllCharacteristics",
    "SizeOfStackReserve",
    "SizeOfStackCommit",
    "SizeOfHeapReserve",
    "SizeOfHeapCommit",
    "LoaderFlags",
    "NumberOfRvaAndSizes",
    "LengthOfPeSections",
    "MeanEntropy",
    "MinEntropy",
    "MaxEntropy",
    "MeanRawSize",
    "MinRawSize",
    "MaxRawSize",
    "MeanVirtualSize",
    "MinVirtualSize",
    "MaxVirtualSize",
    "ImportsNbDLL",
    "ImportsNb",
    "ImportsNbOrdinal",
    "ExportNb",
    "ResourcesNb",
    "ResourcesMeanEntropy",
    "ResourcesMinEntropy",
    "ResourcesMaxEntropy",
    "ResourcesMeanSize",
    "ResourcesMinSize",
    "ResourcesMaxSize",
    "LoadConfigurationSize",
    "VersionInformationSize",
    "DLL",
    "LengthOfInformation",
]

def extract_infos_enhanced(fpath):
    raw = {}
    realdata = {}
    pe = pefile.PE(fpath)
    raw['MD5'] = 1
    raw['SHA-1'] = 1
    raw['SHA-256'] = 1
    raw['SHA-512'] = 1
    raw['e_magic'] = pe.DOS_HEADER.e_magic
    raw['e_cblp'] = (pe.DOS_HEADER.e_cblp)
    raw['e_cp'] = (pe.DOS_HEADER.e_cp)
    raw['e_crlc'] = (pe.DOS_HEADER.e_crlc)
    raw['e_cparhdr'] = (pe.DOS_HEADER.e_cparhdr)
    raw['e_minalloc'] = (pe.DOS_HEADER.e_minalloc)
    raw['e_maxalloc'] = (pe.DOS_HEADER.e_maxalloc)
    raw['e_ss'] = (pe.DOS_HEADER.e_ss)
    raw['e_sp'] = (pe.DOS_HEADER.e_sp)
    raw['c_csum'] = (pe.DOS_HEADER.e_csum)
    raw['e_ip'] = (pe.DOS_HEADER.e_ip)
    raw['c_cs'] = (pe.DOS_HEADER.e_cs)
    raw['e_lfarlc'] = (pe.DOS_HEADER.e_lfarlc)
    raw['e_ovno'] = (pe.DOS_HEADER.e_ovno)
    raw['e_oemid'] = (pe.DOS_HEADER.e_oemid)
    raw['e_oeminfo'] = (pe.DOS_HEADER.e_oeminfo)
    raw['e_lfanew'] = (pe.DOS_HEADER.e_lfanew)
    raw['Machine'] = (pe.FILE_HEADER.Machine)
    raw['SizeOfOptionalHeader'] = (pe.FILE_HEADER.SizeOfOptionalHeader)
    raw['Characteristics'] = (pe.FILE_HEADER.Characteristics)
    raw['Signature'] = (pe.NT_HEADERS.Signature)
    raw['Magic'] = (pe.OPTIONAL_HEADER.Magic)
    raw['MajorLinkerVersion'] = (pe.OPTIONAL_HEADER.MajorLinkerVersion)
    raw['MinorLinkerVersion'] = (pe.OPTIONAL_HEADER.MinorLinkerVersion)
    raw['SizeOfCode'] = (pe.OPTIONAL_HEADER.SizeOfCode)
    raw['SizeOfInitializedData'] = (pe.OPTIONAL_HEADER.SizeOfInitializedData)
    raw['SizeOfUninitializedData'] = (pe.OPTIONAL_HEADER.SizeOfUninitializedData)
    raw['AddressOfEntryPoint'] = (pe.OPTIONAL_HEADER.AddressOfEntryPoint)
    raw['BaseOfCode'] = pe.OPTIONAL_HEADER.BaseOfCode

    try:
        raw['BaseOfData'] = pe.OPTIONAL_HEADER.BaseOfData
    except AttributeError:
        raw['BaseOfData'] = 0

    raw['ImageBase'] = pe.OPTIONAL_HEADER.ImageBase
    raw['SectionAlignment'] = pe.OPTIONAL_HEADER.SectionAlignment
    raw['FileAlignment'] = pe.OPTIONAL_HEADER.FileAlignment
    raw['MajorOperatingSystemVersion'] = pe.OPTIONAL_HEADER.MajorOperatingSystemVersion
    raw['MinorOperatingSystemVersion'] = pe.OPTIONAL_HEADER.MinorOperatingSystemVersion
    raw['MajorImageVersion'] = pe.OPTIONAL_HEADER.MajorImageVersion
    raw['MinorImageVersion'] = pe.OPTIONAL_HEADER.MinorImageVersion
    raw['MajorSubsystemVersion'] = pe.OPTIONAL_HEADER.MajorSubsystemVersion
    raw['MinorSubsystemVersion'] = pe.OPTIONAL_HEADER.MinorSubsystemVersion
    raw['Reserved1'] = pe.OPTIONAL_HEADER.Reserved1
    raw['SizeOfImage'] = pe.OPTIONAL_HEADER.SizeOfImage
    raw['SizeOfHeaders'] = pe.OPTIONAL_HEADER.SizeOfHeaders
    raw['CheckSum'] = pe.OPTIONAL_HEADER.CheckSum
    raw['Subsystem'] = pe.OPTIONAL_HEADER.Subsystem
    raw['DllCharacteristics'] = pe.OPTIONAL_HEADER.DllCharacteristics
    raw['SizeOfStackReserve'] = pe.OPTIONAL_HEADER.SizeOfStackReserve
    raw['SizeOfStackCommit'] = pe.OPTIONAL_HEADER.SizeOfStackCommit
    raw['SizeOfHeapReserve'] = pe.OPTIONAL_HEADER.SizeOfHeapReserve
    raw['SizeOfHeapCommit'] = pe.OPTIONAL_HEADER.SizeOfHeapCommit
    raw['LoaderFlags'] = pe.OPTIONAL_HEADER.LoaderFlags
    raw['NumberOfRvaAndSizes'] = pe.OPTIONAL_HEADER.NumberOfRvaAndSizes
    # Sections
    raw['LengthOfPeSections'] = len(pe.sections)
    
    entropy = list(map(lambda x:x.get_entropy(), pe.sections))
    raw['MeanEntropy'] = sum(entropy)/float(len(entropy))    
    raw['MinEntropy'] = min(entropy)
    raw['MaxEntropy'] = max(entropy)
    
    raw_sizes = list(map(lambda x:x.SizeOfRawData, pe.sections))
    raw['MeanRawSize'] = sum(raw_sizes)/float(len(raw_sizes))
    raw['MinRawSize'] = min(raw_sizes)
    raw['MaxRawSize'] = max(raw_sizes)   

    virtual_sizes = list(map(lambda x:x.Misc_VirtualSize, pe.sections))
    raw['MeanVirtualSize'] = sum(virtual_sizes)/float(len(virtual_sizes))
    raw['MinVirtualSize'] = min(virtual_sizes)
    raw['MaxVirtualSize'] = max(virtual_sizes)

    #Imports
    try:
        raw['ImportsNbDLL'] = len(pe.DIRECTORY_ENTRY_IMPORT)
        imports = sum([x.imports for x in pe.DIRECTORY_ENTRY_IMPORT], [])
        raw['ImportsNb'] = len(imports)
        raw['ImportsNbOrdinal'] = len(list(filter(lambda x:x.name is None, imports)))
    except AttributeError:
        raw['ImportsNbDLL'] = 0
        raw['ImportsNb'] = 0
        raw['ImportsNbOrdinal'] = 0

    #Exports
    try:
        raw['ExportNb'] = len(pe.DIRECTORY_ENTRY_EXPORT.symbols)
    except AttributeError:
        # No export
        raw['ExportNb'] = 0
    #Resources

    resources= get_resources(pe)
    raw['ResourcesNb'] = len(resources)
    if len(resources)> 0:
        entropy = list(map(lambda x:x[0], resources))
        raw['ResourcesMeanEntropy'] = sum(entropy)/float(len(entropy))
        raw['ResourcesMinEntropy'] = min(entropy)
        raw['ResourcesMaxEntropy'] = max(entropy)  
        sizes = list(map(lambda x:x[1], resources))
        raw['ResourcesMeanSize'] = sum(sizes)/float(len(sizes))
        raw['ResourcesMinSize'] = min(sizes)
        raw['ResourcesMaxSize'] = max(sizes)
    else:
        raw['ResourcesNb'] = 0
        raw['ResourcesMeanEntropy'] = 0
        raw['ResourcesMinEntropy'] = 0
        raw['ResourcesMaxEntropy'] = 0
        raw['ResourcesMeanSize'] = 0
        raw['ResourcesMinSize'] = 0
        raw['ResourcesMaxSize'] = 0

    # Load configuration size
    try:
        raw['LoadConfigurationSize'] = pe.DIRECTORY_ENTRY_LOAD_CONFIG.struct.Size
    except AttributeError:
        raw['LoadConfigurationSize'] = 0


    # Version configuration size
    try:
        version_infos = get_version_info(pe)
        raw['VersionInformationSize'] = len(version_infos.keys())
    except AttributeError:
        raw['VersionInformationSize'] = 0
    return raw

    raw["DLL"] = [entryDLL.dll.decode("utf-8") for entryDLL in pe.DIRECTORY_ENTRY_IMPORT ]
    raw["LengthOfInformation"] = len(raw)

    return raw


