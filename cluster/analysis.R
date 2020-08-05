setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(data.table)

tbl <- fread(
  "parameter_analysis_20200630/output.txt", 
  sep = ",",
  header = TRUE,
  colnames = c("p_e", "rho", "phi", "te", "st")
  )

tbl_orig <- tbl


tbl[which(is.na(tbl$te)),]

which(tbl$te > 18e4) / 10000
