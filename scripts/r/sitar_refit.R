# Classical SITAR benchmark (M12): refit sitar for each held-out child and visit count.
#
# Usage: Rscript sitar_refit.R training.csv tasks.csv out.csv df step age_low age_high
#   training.csv  id, age, y            full records of the fold's training children
#   tasks.csv     task, id, visits, age, y   the held-out child's first `visits` visits per task
#   out.csv       task, id, visits, sitar_aphv, status
# The child's age at PHV is the peak of its fitted velocity curve on a grid over [age_low, age_high].

# sitar may live in a user library (installed with lib = "~/R/library").
user_lib <- path.expand(Sys.getenv("YOUTHTWIN_R_LIB", "~/R/library"))
if (dir.exists(user_lib)) .libPaths(c(user_lib, .libPaths()))
suppressPackageStartupMessages(library(sitar))

args <- commandArgs(trailingOnly = TRUE)
training <- read.csv(args[1], stringsAsFactors = FALSE)
tasks <- read.csv(args[2], stringsAsFactors = FALSE)
df <- as.integer(args[4])
grid <- seq(as.numeric(args[6]), as.numeric(args[7]), by = as.numeric(args[5]))

one <- function(rows) {
  data <- rbind(training, rows[, c("id", "age", "y")])
  fit <- tryCatch(sitar(age, y, id, data, df = df), error = function(e) NULL)
  if (is.null(fit)) return(c(NA, "fit_failed"))
  velocity <- tryCatch(
    predict(fit, data.frame(age = grid, id = rows$id[1]), deriv = 1),
    error = function(e) NULL
  )
  if (is.null(velocity) || all(!is.finite(velocity))) return(c(NA, "predict_failed"))
  peak <- which.max(velocity)
  status <- if (peak == 1 || peak == length(grid)) "edge" else "ok"
  c(grid[peak], status)
}

out <- do.call(rbind, lapply(split(tasks, tasks$task), function(rows) {
  result <- one(rows)
  data.frame(
    task = rows$task[1], id = rows$id[1], visits = rows$visits[1],
    sitar_aphv = as.numeric(result[1]), status = result[2]
  )
}))
write.csv(out, args[3], row.names = FALSE)
