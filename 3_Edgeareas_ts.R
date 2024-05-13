topdir = "/Users/samrabin/Library/CloudStorage/Dropbox/2023_NCAR/FATES escaped fire/Lin_edgeareas"
# version = "20240429"
version = "20240506"


# Preamble ----------------------------------------------------------------

library("jsonlite")
library("ggplot2")
library("ggpubr")

setwd(topdir)


# Settings ----------------------------------------------------------------

process_one <- function(input_file, version) {
  lin_original_colors = c(
    "#9E0142" ,
    "#D53E4F",
    "#F46D43" ,
    "#FDAE61",
    "yellowgreen",
    "springgreen3",
    "darkgreen"
  )
  if (version == "20240429") {
    bin_labels = c('<30',
                   '30-100',
                   '100-300',
                   '300-500',
                   '500-1000',
                   '1000-2000',
                   '>2000')
    bin_colors = lin_original_colors
  } else {
    if (version == "20240506") {
      bin_labels = c(
        '<30',
        '30-60',
        '60-90',
        '90-120',
        '120-300',
        '300-500',
        '500-1000',
        '1000-2000',
        '>2000'
      )
    } else {
      stop(sprintf("Version not recognized: %s", version))
    }
    Nbins = length(bin_labels)
    if (Nbins == 7) {
      bin_colors = lin_original_colors
    } else {
      # Use Brewer red-yellow-green instead of Lin's original set of 7 colors
      bin_colors = scales::pal_brewer(palette = "RdYlGn")(Nbins)
      # bin_colors = scales::pal_viridis()(Nbins)
    }
  }
  
  this_theme = theme(
    panel.grid.major = element_blank(),
    legend.position = "right",
    axis.text = element_text(size = 12),
    panel.grid.minor = element_blank()
  )
  
  # Process settings --------------------------------------------------------
  
  color_scale_legend = scale_color_manual("Edge group (m)", values = bin_colors, labels = bin_labels)
  
  # Get total area of each site. Is there any way to automate this?
  if (basename(input_file) == "Edgearea_clean_1.csv") {
    site_area = 354
  } else if (basename(input_file) == "Edgearea_clean_2.csv") {
    site_area = 370
  } else if (basename(input_file) == "Edgearea_clean_3.csv") {
    site_area = 367
  } else if (basename(input_file) == "Edgearea_clean_4.csv") {
    site_area = 98.2
  } else {
    stop("input_file not recognized: can't get site area")
  }
  
  
  # Read CSV ----------------------------------------------------------------
  
  df.forest_bins = read.csv(input_file, header = T)
  
  # Rename column name to something more descriptive
  index_sumarea = which(colnames(df.forest_bins) == "sumarea")
  colnames(df.forest_bins)[index_sumarea] = "Bin.Area"
  
  # Forest area in each bin in each year -------------------------------------------
  
  p = list()
  p[[1]] = ggplot(df.forest_bins
                  ,
                  aes(
                    x = Year,
                    y = Bin.Area,
                    color = as.factor(edge),
                    group = as.factor(edge)
                  )) + geom_point(alpha = 0.7) + geom_line() +
    theme_bw() + ylab(expression(Forest ~ area ~ (km ^ 2))) + xlab("Year") +
    this_theme +
    color_scale_legend
  
  
  # Total forest area in each year ------------------------------------------
  
  total_forest_per_year <- aggregate(Bin.Area ~ Year, data = df.forest_bins, FUN = sum)
  
  p[[2]] = ggplot(total_forest_per_year
                  , aes(x = Year, y = Bin.Area)) + geom_point(alpha = 0.7, color =
                                                                "purple4") +
    geom_line(color = "purple4") +
    theme_bw() + ylab(expression(Total ~ forest ~ area ~ (km ^ 2))) + xlab("Year") +
    this_theme
  
  
  # % of remaining forest in each bin, X axis year --------------------------
  colnames(total_forest_per_year)[2] = "Total.Forest.Area"
  df.merge = merge (df.forest_bins, total_forest_per_year, by = "Year")
  df.merge$Pct.Remaining.Forest = df.merge$Bin.Area / df.merge$Total.Forest.Area * 100
  
  p[[3]] = ggplot(
    df.merge
    ,
    aes(
      x = Year,
      y = Pct.Remaining.Forest,
      color = as.factor(edge),
      group = as.factor(edge)
    )
  ) + geom_point() + geom_line() +
    theme_bw() + ylab("Forest percentage (%)") + xlab("Year") +
    this_theme +
    color_scale_legend
  
  # % of remaining forest in each bin, X axis % deforestation --------------------
  df.merge$deforestation = (1 - df.merge$Total.Forest.Area / site_area) * 100
  p[[4]] = ggplot(
    df.merge
    ,
    aes(
      x = deforestation,
      y = Pct.Remaining.Forest,
      color = as.factor(edge),
      group = as.factor(edge)
    )
  ) + geom_point() + geom_line() +
    theme_bw() + ylab("Forest percentage (%)") + xlab("Cumulative deforestation (%)") +
    this_theme +
    color_scale_legend
  
  
  # Save --------------------------------------------------------------------
  
  output_file = sub("csv", "pdf", input_file)
  print(output_file)
  pdf(output_file, width = 12, height  = 8)
  
  # print() is necessary to save a plot from within a function. See https://cran.r-project.org/doc/FAQ/R-FAQ.html#Why-do-lattice_002ftrellis-graphics-not-work_003f
  print(ggarrange(
    plotlist = p,
    ncol = 2,
    nrow = 2,
    align = "hv"
  ))
  dev.off()
  
  # Make this available outside the function
  df.merge
}

for (i in 1:4) {
  input_file = file.path("inout", version, sprintf("Edgearea_clean_%s.csv", i))
  df.this = process_one(input_file, version)
  if (i == 1) {
    df.all = data.frame(matrix(nrow = 0, ncol = length(colnames(df.this))))
    colnames(df.all) = colnames(df.this)
  }
  df.all = rbind(df.all, df.this)
}

print("Done")