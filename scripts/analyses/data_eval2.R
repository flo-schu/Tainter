library(ggplot2)
library(viridis)
library(ggpubr)
library(ggthemes)
library(svglite)
library(tidyverse)
library(data.table)
library(helfRlein)
rm(list = ls())

########################### DATA IMPORT ########################################

sourcedir <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(sourcedir)
files <- dir("./data")[grep(".csv",dir("./data"))]
j <- 0

for(i in files){
  j <- j+1
  data <- data.table::fread(file = paste0("./data/",i))
  assign(paste("data",j,sep = ""),
         data)
}
rm(data,i,j)


d <- data1 %>% 
  select(-V1) %>% 
  mutate(energyrate = tot.energy / survivaltime) %>% 
  group_by(link.density,exploration,eff) %>% 
  summarise_all(mean) %>% 
  #filter(threshold == 1) %>% 
  ungroup()

hardsum <- function(dat){
  mi <- min(dat)
  ma <- max(dat)
  me <- sort(dat)[round(length(d$link.density)/2L,0)]
  return(c("Min." = mi, "Median" = me, "Max." = ma))
}

closest <- function(x, y){
  return(x[which.min(abs(x - y))] )
}

parz <- c("link.density","exploration","eff")

# rPr <- hardsum(d$link.density)
# rEx <- hardsum(d$exploration)
# rEf <- hardsum(d$eff)
rPr <- c( 0   , 0.01  , 0.1   )
rEx <- c( 0   , closest(d$exploration, 0.02)  , closest(d$exploration, 0.8))
rEf <- c( 1   , 1.1 , 1.3 )


pars <- list(rPr = rPr, rEx = rEx, rEf = rEf)
dlist <- list()
plist <- list()
ilist <- list()

date <- "190222"
index <- c("survivaltime","tot.energy","wellbeing","energyrate")

for (l in index) {
  for (i in 1:length(pars)) {
    dlist[[i]] <- list()
    ind     <- quo(!!sym(l))
    rowpar  <- quo(!!sym(parz[i]))
    plopar1 <- quo(!!sym(parz[-i][1]))
    plopar2 <- quo(!!sym(parz[-i][2]))
    labz <- as.data.frame(t(paste(parz[i],"=",round(pars[[i]],2))))
    
    for (j in 1:length(rPr)) {
      dlist[[i]][[j]] <- d %>% 
        filter(!! rowpar == pars[[i]][j])
      plist[[(i-1)*length(pars)+j]] <- 
        ggplot(dlist[[i]][[j]], 
               aes(x    = !! plopar1, 
                   y    = !! plopar2, 
                   fill = !! ind))+
        geom_tile()+
        theme_few() + 
        scale_fill_viridis(limits = c(0,max(d[,l])) ,discrete = FALSE)+
        geom_label(data = labz, 
                   aes_string(x=Inf,y=Inf,label=paste0("V",j), hjust = 1, vjust = 1), 
                   inherit.aes = FALSE, alpha = .5)
      
    }
  }
  ilist[[l]] <- plist
}


folders <- dir("./plots")[grep(date,dir("./plots"))]
newfolder <- paste0("./plots/",date,"_",length(folders))
dir.create(newfolder)


for(i in 1:length(index)){
  g <- ggarrange(plotlist = ilist[[i]], ncol = 3, nrow = 3, 
            common.legend = TRUE, legend = "right")
  
  ggsave(filename = paste0(newfolder,"/",index[i],".pdf"),
         plot = g,
         width=16, height = 10, dpi = 300)
}

