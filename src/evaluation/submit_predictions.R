install.packages("devtools")
require(devtools)
library(devtools)
install_github("Weecology/NeonTreeEvaluation_package")
devtools::install_github("nx10/httpgd")

library(NeonTreeEvaluation)
download(savedir="data/evaluation")
list_rgb()

eval_url <- zenodo_url(concept_rec_id = 3723356)

# read in benchmark_predictions
benchmark_predictions <- read.csv("src/evaluation/benchmark_predictions.csv")

# evaluate crown predictions using evaluate_image_crowns
evaluate_image_crowns(predictions = benchmark_predictions, project = T, show = F, summarize = T)
