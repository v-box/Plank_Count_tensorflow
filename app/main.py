import nest_asyncio
nest_asyncio.apply()

import uvicorn
from typing import List
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse,StreamingResponse

from utils import load_model,infer_image
import os
cwd = os.getcwd()

app = FastAPI()

detect_fn,category_index = load_model()


@app.post("/staff_detection/")
async def create_upload_files(files: List[UploadFile] = File(...)):
    file_list = [file for file in files]
    im_lst = []
    non_im = []
    for file in file_list:
        if file.content_type=='image/png' or file.content_type=='image/jpeg':    
            im_lst.append(file) 
            image_file = await file.read()
            file_location = f"{cwd}/{file.filename}"
            with open(file_location, "wb+") as file_object:
                file_object.write(image_file)
            count,img = infer_image(file_location,detect_fn,category_index)
            os.remove(file_location)
            img.save(f'boxed_{file.filename}')
            img_p = str(os.getcwd())+f'/boxed_{file.filename}'
            #img.save(f'Boxed_{file.filename}')
        else:
            non_im.append(file.filename)
        
        def iterfile():
          with open(img_p, mode="rb") as file_like:  
              yield from file_like  

        return StreamingResponse(iterfile(), media_type=str(file.content_type))
        #return {count,img} #{f'Count Of Planks in {file.filename} is ':str(count)} #count,im_lst {count,img}



@app.get("/")
async def main():
    content = """
<body>
<form action="/staff_detection/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)

# When Dockerising an App, Comment host and uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8080, debug=True)
