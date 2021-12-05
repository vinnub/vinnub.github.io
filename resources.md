---
layout: default 
nav_order: 3
title: Resources 
---
# Resources
{: .no_toc }
Here, I have a list of resources for different kinds of things that I learned in grad school including things like how to set up Jupyter on cloud, how to download NGS data efficiently etc. I still come back to these for a refresher sometimes. If you are just getting started in the field of bioinformatics and data science, some of these might be helpful.

---
1. TOC
{:toc}
---


### Linux
New to linux? Start with this simple [tutorial](https://ryanstutorials.net/linuxtutorial/) that goes over the basics in short amount of time. Ryan also has tutorials on other topics such as HTML, CSS etc.  Here are two cheatsheets that I sometimes refer to - [Cheatsheet 1](https://wiki.bits.vib.be/index.php/Linux_Beginner%27s_Shell_Cheat_page) and [Cheatsheet 2](https://wiki.bits.vib.be/index.php/The_practical_command_line_cheat_sheet). 

### Git 
There are a lot of resources for Git out there and I don't have any favorites. My suggestion would be to pick one and get started. Start with the 
basics and learn more as you need to instead of trying to learn it all at once. 
[Here](https://kbroman.org/github_tutorial/) is a simple guide to get started.  

### Regular Expressions
[Here](https://www.youtube.com/watch?v=DRR9fOXkfRE&feature=youtu.be) is a quick youtube video to get started with regular expressions. 


### Python (Import statements and decorators)
Having the right directory structure for your code can be really helpful and make the code much easier to understand. Understanding how import statements work (the difference between relative and absolute imports etc.) in Python can make structuring the repo a lot easier and more maintainable. This [page](https://chrisyeh96.github.io/2017/08/08/definitive-guide-python-imports.html) explains how import statements work. A highly recommended read. Another thing that I really love is Python decorators. Read [this](https://realpython.com/primer-on-python-decorators/) to get started on those. 


### Jupyter on Google Cloud Platform  
Jupyter notebooks are amazing for exploratory data analysis and much more. For bigger datasets and high compute tasks you may want to run the notebook on a cloud server. [This post](https://jeffdelaney.me/blog/running-jupyter-notebook-google-cloud-platform/) will get you started if you use Google Cloud. If you want GPU-enabled notebooks on AWS, have a look at [this](https://course.fast.ai/start_aws).

### Downloading bulk NGS Data efficiently
Here are two posts that really helped me when I wanted to download hundreds of immunosequencing datasets for one of my PhD projects ([Post 1](https://www.michaelgerth.net/news--blog/how-to-efficiently-bulk-download-ngs-data-from-sequence-read-databases) and [Post 2](https://www.biostars.org/p/325010/)). The download speeds were much faster than using the SRA toolkit. 