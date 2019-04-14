
sourcedir <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(sourcedir)
library(tidyverse)
library(data.table)

dat <- fread("./data/data_2019-02-22-1038.csv")

data <- dat %>% 
  filter(link.density == .05) %>% 
  filter(threshold == 1) %>% 
  filter(exploration < .011 & exploration > .009)

fwrite(data, file = "./results/model/20190327_1730/filtered_data.csv")
