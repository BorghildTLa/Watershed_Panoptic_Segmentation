
def getWsi(path): #imports a WSI

  OPENSLIDE_PATH = r"C:\Users\borgh\Downloads\openslide-bin-4.0.0.6-windows-x64\openslide-bin-4.0.0.6-windows-x64\bin"

  if hasattr(os, 'add_dll_directory'):
      # Windows
      with os.add_dll_directory(OPENSLIDE_PATH):
          import openslide
  else:
      import openslide

  wsi = openslide.OpenSlide(path)
  return wsi
