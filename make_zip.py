import os
import zipfile

def zipdir(path, ziph):
    # Make sure we add the base folder 'chatbot' first
    base_folder = os.path.basename(os.path.normpath(path))
    ziph.write(path, base_folder + '/')
    
    for root, dirs, files in os.walk(path):
        for d in dirs:
            dir_path = os.path.join(root, d)
            rel_path = os.path.relpath(dir_path, os.path.dirname(path))
            # Use forward slashes
            rel_path = rel_path.replace('\\\\', '/') + '/'
            ziph.write(dir_path, rel_path)
            
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, os.path.dirname(path))
            rel_path = rel_path.replace('\\\\', '/')
            ziph.write(file_path, rel_path)

zipf = zipfile.ZipFile('../plugin/new/chatbot_fixed.zip', 'w', zipfile.ZIP_DEFLATED)
zipdir('tmp/chatbot_plugin/chatbot', zipf)
zipf.close()
