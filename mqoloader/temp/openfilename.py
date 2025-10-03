#ファイルダイアログ
import ctypes
import ctypes.wintypes as wintypes

LPOFNHOOKPROC = ctypes.c_voidp
LPCTSTR = LPTSTR = ctypes.c_wchar_p

class OPENFILENAME(ctypes.Structure):
	_fields_ = [("lStructSize", wintypes.DWORD),
	("hwndOwner", wintypes.HWND),("hInstance", wintypes.HINSTANCE),
	("lpstrFilter", LPCTSTR),("lpstrCustomFilter", LPTSTR),
	("nMaxCustFilter", wintypes.DWORD),("nFilterIndex", wintypes.DWORD),
	("lpstrFile", LPTSTR),("nMaxFile", wintypes.DWORD),
	("lpstrFileTitle", LPTSTR),("nMaxFileTitle", wintypes.DWORD),
	("lpstrInitialDir", LPCTSTR),("lpstrTitle", LPCTSTR),
	("flags", wintypes.DWORD),("nFileExtension", wintypes.WORD),
	("lpstrDefExt", LPCTSTR),("lCustData", wintypes.LPARAM),
	("lpfnHook", LPOFNHOOKPROC),("lpTemplateName", LPCTSTR)]

get_open_file_name = ctypes.windll.comdlg32.GetOpenFileNameW
get_save_file_name = ctypes.windll.comdlg32.GetSaveFileNameW

OFN_ENABLESIZING=0x00800000
OFN_PATHMUSTEXIST=0x00000800
OFN_OVERWRITEPROMPT=0x00000002
OFN_NOCHANGEDIR=0x00000008
MAX_PATH=1024

def _build_ofn(title, default_extension, filter_string, file_buffer):
	filter = ctypes.create_unicode_buffer(filter_string)
	ofn = OPENFILENAME()
	ofn.lStructSize = ctypes.sizeof(OPENFILENAME)
	ofn.lpstrTitle = title
	ofn.lpstrFile = ctypes.cast(file_buffer, LPTSTR)
	ofn.nMaxFile = MAX_PATH
	ofn.lpstrDefExt = default_extension
	ofn.lpstrFilter = ctypes.cast(filter, ctypes.c_wchar_p)
	ofn.Flags = OFN_ENABLESIZING | OFN_PATHMUSTEXIST \
		| OFN_OVERWRITEPROMPT | OFN_NOCHANGEDIR
	return ofn

def open_file_name(
		title,default_extension,filter_string,initial_path):
	if initial_path is None:
		initial_path = ""
	filter_string = filter_string.replace("|", "\0")
	file_buffer = ctypes.create_unicode_buffer(initial_path, MAX_PATH)
	ofn=_build_ofn(title,default_extension,filter_string,file_buffer)
	
	if get_open_file_name(ctypes.byref(ofn)):
		return file_buffer.value
	return None
