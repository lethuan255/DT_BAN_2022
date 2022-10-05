import os
import hashlib
import pefile
import peutils
import json

def Convert(lst):
    res_dct = {lst[i]: lst[i + 1] for i in range(0, len(lst), 2)}
    return res_dct

def get_entropy(data):
    if len(data) == 0:
        return 0.0
    occurences = array.array('L', [0] * 256)
    for x in data:
        occurences[x if isinstance(x, int) else ord(x)] += 1

    entropy = 0
    for x in occurences:
        if x:
            p_x = float(x) / len(data)
            entropy -= p_x * math.log(p_x, 2)

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
                                data = pe.get_data(resource_lang.data.struct.OffsetToData,
                                                   resource_lang.data.struct.Size)
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





def getFullinformation(file_path):
	print(file_path)
	#file_path.replace("/", "//")

	data = []
	try:
		pe = pefile.PE(file_path)

		raw = pe.write()
		
		#print((pe.dump_info()))
		#'''
		#data.append(pe.DOS_HEADER.e_magic)
		data.append("FileName")
		data.append(os.path.basename(file_path))
		
		data.append("MD5")
		data.append(hashlib.md5(raw).hexdigest())

		data.append("SHA-1")
		try:
			data.append(1)
		except:
			data.append(0)


		data.append("SHA-256")
		try:
			data.append(1)
		except:
			data.append(0)

		data.append("SHA-512")
		try:
			data.append(1)
		except:
			data.append(0)


		data.append("e_magic")
		data.append(pe.DOS_HEADER.e_magic)

		data.append("e_cblp")
		data.append(pe.DOS_HEADER.e_cblp)
		
		data.append("e_cp")
		data.append(pe.DOS_HEADER.e_cp)

		data.append("e_crlc")
		data.append(pe.DOS_HEADER.e_crlc)

		data.append("e_cparhdr")
		data.append(pe.DOS_HEADER.e_cparhdr)

		data.append("e_minalloc")
		data.append(pe.DOS_HEADER.e_minalloc)

		data.append("e_maxalloc")
		data.append(pe.DOS_HEADER.e_maxalloc)

		data.append("e_ss")
		data.append(pe.DOS_HEADER.e_ss)

		data.append("e_sp")
		data.append(pe.DOS_HEADER.e_sp)

		data.append("c_csum")
		data.append(pe.DOS_HEADER.e_csum)

		data.append("e_ip")
		data.append(pe.DOS_HEADER.e_ip)

		data.append("c_cs")
		data.append(pe.DOS_HEADER.e_cs)

		data.append("e_lfarlc")
		data.append(pe.DOS_HEADER.e_lfarlc)

		data.append("e_ovno")
		data.append(pe.DOS_HEADER.e_ovno)
		#data.append(pe.DOS_HEADER.e_res)

		data.append("e_oemid")
		data.append(pe.DOS_HEADER.e_oemid)

		data.append("e_oeminfo")
		data.append(pe.DOS_HEADER.e_oeminfo)
		#data.append(pe.DOS_HEADER.e_res2)

		data.append("e_lfanew")
		data.append(pe.DOS_HEADER.e_lfanew)

		data.append("Machine")
		data.append(pe.FILE_HEADER.Machine)

		data.append("SizeOfOptionalHeader")
		data.append(pe.FILE_HEADER.SizeOfOptionalHeader)

		data.append("Characteristics")
		data.append(pe.FILE_HEADER.Characteristics)

		data.append("Signature")
		data.append(pe.NT_HEADERS.Signature)

		data.append("Magic")
		data.append(pe.OPTIONAL_HEADER.Magic)

		data.append("MajorLinkerVersion")
		data.append(pe.OPTIONAL_HEADER.MajorLinkerVersion)

		data.append("MinorLinkerVersion")
		data.append(pe.OPTIONAL_HEADER.MinorLinkerVersion)

		data.append("SizeOfCode")
		data.append(pe.OPTIONAL_HEADER.SizeOfCode)

		data.append("SizeOfInitializedData")
		data.append(pe.OPTIONAL_HEADER.SizeOfInitializedData)

		data.append("SizeOfUninitializedData")
		data.append(pe.OPTIONAL_HEADER.SizeOfUninitializedData)
		
		data.append("AddressOfEntryPoint")
		data.append(pe.OPTIONAL_HEADER.AddressOfEntryPoint)

		data.append("BaseOfCode")
		data.append(pe.OPTIONAL_HEADER.BaseOfCode)

		data.append("BaseOfData")
		try:
			data.append(pe.OPTIONAL_HEADER.BaseOfData)
		except AttributeError:
			data.append(0)


		data.append("ImageBase")
		data.append(pe.OPTIONAL_HEADER.ImageBase)

		data.append("SectionAlignment")
		data.append(pe.OPTIONAL_HEADER.SectionAlignment)

		data.append("FileAlignment")
		data.append(pe.OPTIONAL_HEADER.FileAlignment)

		data.append("MajorOperatingSystemVersion")
		data.append(pe.OPTIONAL_HEADER.MajorOperatingSystemVersion)

		data.append("MinorOperatingSystemVersion")
		data.append(pe.OPTIONAL_HEADER.MinorOperatingSystemVersion)

		data.append("MajorImageVersion")
		data.append(pe.OPTIONAL_HEADER.MajorImageVersion)

		data.append("MinorImageVersion")
		data.append(pe.OPTIONAL_HEADER.MinorImageVersion)

		data.append("MajorSubsystemVersion")
		data.append(pe.OPTIONAL_HEADER.MajorSubsystemVersion)

		data.append("MinorSubsystemVersion")
		data.append(pe.OPTIONAL_HEADER.MinorSubsystemVersion)

		data.append("Reserved1")
		data.append(pe.OPTIONAL_HEADER.Reserved1)

		data.append("SizeOfImage")
		data.append(pe.OPTIONAL_HEADER.SizeOfImage)

		data.append("SizeOfHeaders")
		data.append(pe.OPTIONAL_HEADER.SizeOfHeaders)

		data.append("CheckSum")
		data.append(pe.OPTIONAL_HEADER.CheckSum)

		data.append("Subsystem")
		data.append(pe.OPTIONAL_HEADER.Subsystem)

		data.append("DllCharacteristics")
		data.append(pe.OPTIONAL_HEADER.DllCharacteristics)

		data.append("SizeOfStackReserve")
		data.append(pe.OPTIONAL_HEADER.SizeOfStackReserve)

		data.append("SizeOfStackCommit")
		data.append(pe.OPTIONAL_HEADER.SizeOfStackCommit)

		data.append("SizeOfHeapReserve")
		data.append(pe.OPTIONAL_HEADER.SizeOfHeapReserve)

		data.append("SizeOfHeapCommit")
		data.append(pe.OPTIONAL_HEADER.SizeOfHeapCommit)

		data.append("LoaderFlags")
		data.append(pe.OPTIONAL_HEADER.LoaderFlags)

		data.append("NumberOfRvaAndSizes")
		data.append(pe.OPTIONAL_HEADER.NumberOfRvaAndSizes)

		data.append("LengthOfPeSections")
		data.append(len(pe.sections))


		entropy = [x.get_entropy() for x in pe.sections]
		data.append("MeanEntropy")
		data.append(sum(entropy) / float(len(entropy)))
		
		data.append("MinEntropy")
		data.append(min(entropy))
		
		data.append("MaxEntropy")
		data.append(max(entropy))


		raw_sizes = [x.SizeOfRawData for x in pe.sections]
		data.append("MeanRawSize")
		data.append(sum(raw_sizes) / float(len(raw_sizes)))
		
		data.append("MinRawSize")
		data.append(min(raw_sizes))
		
		data.append("MaxRawSize")
		data.append(max(raw_sizes))


		virtual_sizes = [x.Misc_VirtualSize for x in pe.sections]
		data.append("MeanVirtualSize")
		data.append(sum(virtual_sizes) / float(len(virtual_sizes)))
		data.append("MinVirtualSize")
		data.append(min(virtual_sizes))
		data.append("MaxVirtualSize")
		data.append(max(virtual_sizes))
	   
		data.append("ImportsNbDLL")
		try:
			data.append(len(pe.DIRECTORY_ENTRY_IMPORT))
		except:
			data.append(0)
		data.append("ImportsNb")
		try:
			imports = sum([x.imports for x in pe.DIRECTORY_ENTRY_IMPORT], [])
			data.append(len(imports))
		except:
			data.append(0)
		data.append("ImportsNbOrdinal")
		try:
			data.append(len([x for x in imports if x.name is None]))
		except:
			data.append(0)

		data.append("ExportNb")
		try:
			data.append(len(pe.DIRECTORY_ENTRY_EXPORT.symbols))
		except AttributeError:
	        # No export
			data.append(0)


		resources = get_resources(pe)
		data.append("ResourcesNb")
		data.append(len(resources))


		if len(resources) > 0:
			entropy = [x[0] for x in resources]
			data.append("ResourcesMeanEntropy")
			data.append(sum(entropy) / float(len(entropy)))

			data.append("ResourcesMinEntropy")
			data.append(min(entropy))

			data.append("ResourcesMaxEntropy")
			data.append(max(entropy))

			sizes = [x[1] for x in resources]

			data.append("ResourcesMeanSize")
			data.append(sum(sizes) / float(len(sizes)))

			data.append("ResourcesMinSize")
			data.append(min(sizes))

			data.append("ResourcesMaxSize")
			data.append(max(sizes))
		else:
			data.append("ResourcesMeanEntropy")
			data.append(0)

			data.append("ResourcesMinEntropy")
			data.append(0)

			data.append("ResourcesMaxEntropy")
			data.append(0)

			data.append("ResourcesMeanSize")
			data.append(0)

			data.append("ResourcesMinSize")
			data.append(0)

			data.append("ResourcesMaxSize")
			data.append(0)


		data.append("LoadConfigurationSize")
		try:
			data.append(pe.DIRECTORY_ENTRY_LOAD_CONFIG.struct.Size)
		except AttributeError:
			data.append(0)

	
		data.append("VersionInformationSize")
		try:
			version_infos = get_version_info(pe)
			data.append(len(list(version_infos.keys())))
		except AttributeError:
			data.append(0)

		fileDll = []
		for entryDLL in pe.DIRECTORY_ENTRY_IMPORT:
			fileDll.append(entryDLL.dll.decode("utf-8"))

		data.append("DLL")
		data.append(len(fileDll))

		data.append("LengthOfInformation")
		data.append(len(data))


		data = Convert(data)
		json_object = json.dumps(data)

		with open("trichXuat.json", "a") as outfile:
			outfile.write(json_object)
			outfile.write("\n")

	except Exception as exc:
		print(exc)

	

def QuetJSON(folderPath, namelist):
	for file in os.listdir(folderPath):
		if(os.path.isdir(os.path.join(folderPath, file))):
			 QuetJSON(os.path.join(folderPath, file), namelist)
		else:
			namelist.append(str(os.path.join(folderPath, file)))
			getFullinformation(os.path.join(folderPath, file))


def main():
	print("Do stuff")



if __name__ == '__main__':
	main()
	#getFullinformation("E://TestPEFile//0b65c9b63092c96fc737ef39a1f05e437d5bbe57f0ca2bc634a14a34a92ebcb4.exe")

	#getFullinformation("E://TestPEFile//2d901bf0cb31995d596329a8406471c6e82671811c0d16255cfa02154e6dd90b.exe")

	#getFullinformation("E://TestPEFile//GenshinImpact.exe")

'''
  DOS_HEADER
    NT_HEADERS
    FILE_HEADER
    OPTIONAL_HEADER

 	DIRECTORY_ENTRY_IMPORT (list of ImportDescData instances)
    DIRECTORY_ENTRY_EXPORT (ExportDirData instance)
    DIRECTORY_ENTRY_RESOURCE (ResourceDirData instance)
    DIRECTORY_ENTRY_DEBUG (list of DebugData instances)
    DIRECTORY_ENTRY_BASERELOC (list of BaseRelocationData instances)
    DIRECTORY_ENTRY_TLS
    DIRECTORY_ENTRY_BOUND_IMPORT (list of BoundImportData instances)


 	DIRECTORY_ENTRY
    IMAGE_CHARACTERISTICS
    SECTION_CHARACTERISTICS
    DEBUG_TYPE
    SUBSYSTEM_TYPE
    MACHINE_TYPE
    RELOCATION_TYPE
    RESOURCE_TYPE
    LANG
    SUBLANG




'''