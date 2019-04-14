library(ggplot2)
library(viridis)
library(ggpubr)
library(ggthemes)
library(svglite)
library(tidyverse)
#################################### IMPORT FUNCTIONS ##########################
my_plots <- function(mydata, myx, mycolor, myalpha){
  mydata <- cbind(mydata,mesuratio = mydata$maxenergy/mydata$survivaltime)
  
  p1 <- ggplot(data = mydata, 
               mapping = aes_string(x = myx, 
                                    y = "maxenergy"))+
    geom_point(aes_string(color = mycolor), alpha = myalpha)
    #geom_smooth(aes_string(color = mycolor))
  
  p2 <- ggplot(data = mydata, 
               mapping = aes_string(x = myx, 
                                    y = "survivaltime"))+
    geom_point(aes_string(color = mycolor), alpha = myalpha)
    #geom_smooth(aes_string(color = mycolor))
  
  p3 <- ggplot(data = mydata, 
               mapping = aes_string(x = myx, 
                                    y = "mesuratio"))+
    geom_point(aes_string(color = mycolor), alpha = myalpha)
    #geom_smooth(aes_string(color = mycolor))
  try(p4 <- ggplot(data = mydata, 
                   mapping = aes_string(x = myx, 
                                        y = "wellbeing"))+
        geom_point(aes_string(color = mycolor), alpha = myalpha)
        #geom_smooth(aes_string(color = mycolor))
  )
  
  return(list(p1,p2,p3,try(p4)))
}
make_plots <- function(par1, par2, alph = 0.75, discrete = FALSE){
  new_plots <- list()
  for(i in 1:length(files)){
    p <- my_plots(get(paste("data",i,sep="")),par1,par2, myalpha = alph)
    p <- lapply(p, function(i) i + scale_color_viridis(discrete = discrete) + theme_few())
    new_plots[[i]] <- p
  }
  return(new_plots)
}
################################################################################

#################################### IMPORT FILES ##############################
sourcedir <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(sourcedir)
files <- dir("./data")[grep(".csv",dir("./data"))]
j <- 0
for(i in files){
  j <- j+1
  data <- read.csv(paste("./data/",i,sep=""))
  assign(paste("data",j,sep = ""),
         data)
}
rm(data,i,j)
################################################################################


################################################################################
# Hier k?nnen die zu plottenden Vairablen ge?ndert werden!!
new_plots <- make_plots(par1 = "exploration", par2 = "links", discrete = FALSE)
################################################################################

for(i in 1:length(new_plots)){
  assign(paste("p",i,sep=""),new_plots[[i]])
}
rm(new_plots)

################################################################################
# Drucken der Plots. Wenn nur das Plot in R Studio angezeigt werden soll, dann
# einfach die ggarrange funktion ausf?hren.
ggarrange(plotlist = p2,ncol = 4, nrow=1,labels = "AUTO",
          common.legend = TRUE, legend = "bottom") 
 ################################################################################

data2 %>% 
  filter(links < 7) %>% 

  ggplot(aes(x = eff, y = survivaltime)) +
  geom_point(alpha = 0.1)


data4 %>% 
  filter(survivaltime < 100) %>% 
  select(exploration) %>% 
  summarise(min = min(exploration))
  filter(exploration < 0.001) %>% 
  filter(eff < 1.5)

  
colnames(data1)

summary(data1)


apply(data1, 2, unique)

