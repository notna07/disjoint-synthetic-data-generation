# Description: Script for making baselines
# Author: Anton D. Lautrup
# Date: 28-09-2023

library(synthpop)

synthpop_random <- function(input_csv, output_name, num_to_generate) {
    data <- read.csv(input_csv)

    mysyn <- syn(data,
                method = "cart",
                k = num_to_generate,
                minnumlevels = 3,
                print.flag = FALSE
                )

    write.syn(mysyn,
            output_name,
            filetype = "csv",
            save.complete = FALSE,
            extended.info = FALSE,
            )
}

# Command-line arguments
args <- commandArgs(trailingOnly = TRUE)
input_file <- args[1]
output_file <- args[2]
num_to_generate <- as.numeric(args[3])

# Call the function with arguments
synthpop_random(input_file, output_file, num_to_generate)