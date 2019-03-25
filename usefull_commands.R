library(blogdown)
blogdown::serve_site()
blogdown:::new_post_addin()

![](https://media.giphy.com/media/xUOxf7XfmpxuSode1O/giphy.gif)

blogdown::hugo_build(local=F)

#To comment
# {{ % ... %}}
#or
#{{ % ... %}}


library(namer)

knitr::opts_knit$set(root.dir = normalizePath(
  "~/R/Blog TwoSidesData/TwoSidesData/content/post"))

setwd("~/R/Blog TwoSidesData/TwoSidesData/content/post")

name_chunks('')
