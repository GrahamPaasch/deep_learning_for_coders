#!/usr/bin/env python
# coding: utf-8

# In[1]:


#hide
from utils import *
from fastai.vision.widgets import *


# # From Model to Production

# ## The Practice of Deep Learning

# ### Starting Your Project

# ### The State of Deep Learning

# #### Computer vision

# #### Text (natural language processing)

# #### Combining text and images

# #### Tabular data

# #### Recommendation systems

# #### Other data types

# ### The Drivetrain Approach

# ## Gathering Data

# In[2]:


images = []

# grizzly bear data
file = 'grizzly_bears.json'
with open(file) as f:
    data = json.load(f)

grizzly_image_links = []
for result in data["value"]:
    grizzly_image_links.append(result["contentUrl"])

images.append(grizzly_image_links)

# black bear data
file = 'black_bears.json'
with open(file) as f:
    data = json.load(f)

black_image_links = []
for result in data["value"]:
    black_image_links.append(result["contentUrl"])

images.append(black_image_links)

# teddy bear data
file = 'teddy_bears.json'
with open(file) as f:
    data = json.load(f)

teddy_image_links = []
for result in data["value"]:
    teddy_image_links.append(result["contentUrl"])

images.append(teddy_image_links)

example = images[0][7]


# In[3]:


dest = 'images/grizzly.jpg'
download_url(example, dest, overwrite=True)
help(download_url)


# In[4]:


im = Image.open(dest)
im.to_thumb(128,128)


# In[5]:


bear_types = 'grizzly','black','teddy'
path = Path('bears')


# In[6]:


if not path.exists():
    path.mkdir()
    a = 0
    for o in bear_types:
        dest = (path/o)
        dest.mkdir(exist_ok=True)
        for index, link in enumerate(images[a]):
            download_url(link, str(dest) + "/" + str(index) + ".jpg", overwrite=True)
        a += 1


# In[7]:


fns = get_image_files(path)
fns


# In[8]:


failed = verify_images(fns)
failed


# In[9]:


failed.map(Path.unlink);


# ### Sidebar: Getting Help in Jupyter Notebooks

# ### End sidebar

# ## From Data to DataLoaders

# In[15]:


bears = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128))


# In[16]:


dls = bears.dataloaders(path)


# In[17]:


dls.valid.show_batch(max_n=4, nrows=1)


# In[18]:


bears = bears.new(item_tfms=Resize(128, ResizeMethod.Squish))
dls = bears.dataloaders(path)
dls.valid.show_batch(max_n=4, nrows=1)


# In[19]:


bears = bears.new(item_tfms=Resize(128, ResizeMethod.Pad, pad_mode='zeros'))
dls = bears.dataloaders(path)
dls.valid.show_batch(max_n=4, nrows=1)


# In[21]:


bears = bears.new(item_tfms=RandomResizedCrop(128, min_scale=0.3))
dls = bears.dataloaders(path)
dls.train.show_batch(max_n=4, nrows=1, unique=True)


# ### Data Augmentation

# In[22]:


bears = bears.new(item_tfms=Resize(128), batch_tfms=aug_transforms(mult=2))
dls = bears.dataloaders(path)
dls.train.show_batch(max_n=8, nrows=2, unique=True)


# ## Training Your Model, and Using It to Clean Your Data

# In[23]:


bears = bears.new(
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
    batch_tfms=aug_transforms())
dls = bears.dataloaders(path)


# In[24]:


learn = cnn_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(4)


# In[25]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()


# In[26]:


interp.plot_top_losses(5, nrows=1)


# In[27]:


cleaner = ImageClassifierCleaner(learn)
cleaner


# In[28]:


#hide
# for idx in cleaner.delete(): cleaner.fns[idx].unlink()
# for idx,cat in cleaner.change(): shutil.move(str(cleaner.fns[idx]), path/cat)


# ## Turning Your Model into an Online Application

# ### Using the Model for Inference

# In[29]:


learn.export()


# In[30]:


path = Path()
path.ls(file_exts='.pkl')


# In[31]:


learn_inf = load_learner(path/'export.pkl')


# In[32]:


learn_inf.predict('images/grizzly.jpg')


# In[33]:


learn_inf.dls.vocab


# ### Creating a Notebook App from the Model

# In[81]:


btn_upload = widgets.FileUpload()
btn_upload


# In[82]:


#hide
# For the book, we can't actually click an upload button, so we fake it
# btn_upload = SimpleNamespace(data = ['images/grizzly.jpg'])


# In[83]:


img = PILImage.create(btn_upload.data[-1])
img


# In[84]:


out_pl = widgets.Output()
out_pl.clear_output()
with out_pl: display(img.to_thumb(128,128))
out_pl


# In[85]:


pred,pred_idx,probs = learn_inf.predict(img)


# In[86]:


lbl_pred = widgets.Label()
lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'
lbl_pred


# In[87]:


btn_run = widgets.Button(description='Classify')
btn_run


# In[88]:


def on_click_classify(change):
    img = PILImage.create(btn_upload.data[-1])
    out_pl.clear_output()
    with out_pl: display(img.to_thumb(128,128))
    pred,pred_idx,probs = learn_inf.predict(img)
    lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'

btn_run.on_click(on_click_classify)


# In[89]:


#hide
#Putting back btn_upload to a widget for next cell
btn_upload = widgets.FileUpload()


# In[90]:


VBox([widgets.Label('Select your bear!'), 
      btn_upload, btn_run, out_pl, lbl_pred])


# ### Turning Your Notebook into a Real App

# In[95]:


#hide
#!pip install voila
get_ipython().system('jupyter serverextension enable voila')


# ### Deploying your app

# ## How to Avoid Disaster

# ### Unforeseen Consequences and Feedback Loops

# ## Get Writing!

# ## Questionnaire

# 1. Provide an example of where the bear classification model might work poorly in production, due to structural or style differences in the training data.
# 1. Where do text models currently have a major deficiency?
# 1. What are possible negative societal implications of text generation models?
# 1. In situations where a model might make mistakes, and those mistakes could be harmful, what is a good alternative to automating a process?
# 1. What kind of tabular data is deep learning particularly good at?
# 1. What's a key downside of directly using a deep learning model for recommendation systems?
# 1. What are the steps of the Drivetrain Approach?
# 1. How do the steps of the Drivetrain Approach map to a recommendation system?
# 1. Create an image recognition model using data you curate, and deploy it on the web.
# 1. What is `DataLoaders`?
# 1. What four things do we need to tell fastai to create `DataLoaders`?
# 1. What does the `splitter` parameter to `DataBlock` do?
# 1. How do we ensure a random split always gives the same validation set?
# 1. What letters are often used to signify the independent and dependent variables?
# 1. What's the difference between the crop, pad, and squish resize approaches? When might you choose one over the others?
# 1. What is data augmentation? Why is it needed?
# 1. What is the difference between `item_tfms` and `batch_tfms`?
# 1. What is a confusion matrix?
# 1. What does `export` save?
# 1. What is it called when we use a model for getting predictions, instead of training?
# 1. What are IPython widgets?
# 1. When might you want to use CPU for deployment? When might GPU be better?
# 1. What are the downsides of deploying your app to a server, instead of to a client (or edge) device such as a phone or PC?
# 1. What are three examples of problems that could occur when rolling out a bear warning system in practice?
# 1. What is "out-of-domain data"?
# 1. What is "domain shift"?
# 1. What are the three steps in the deployment process?

# ### Further Research

# 1. Consider how the Drivetrain Approach maps to a project or problem you're interested in.
# 1. When might it be best to avoid certain types of data augmentation?
# 1. For a project you're interested in applying deep learning to, consider the thought experiment "What would happen if it went really, really well?"
# 1. Start a blog, and write your first blog post. For instance, write about what you think deep learning might be useful for in a domain you're interested in.

# In[ ]:




