# Set working directory
setwd("/path/to/project/PCoA_analysis")

### 1. Load input data
# OTU abundance table (samples × OTUs)
otu <- read.delim("data/MSB_otu.txt",
                  row.names = 1, sep = "\t",
                  stringsAsFactors = FALSE,
                  check.names = FALSE)

# Transpose OTU table to OTUs × samples
otu <- data.frame(t(otu))

# Sample metadata file (first column must match sample IDs)
group <- read.delim("metadata/MAB_group.txt",
                    sep = "\t",
                    stringsAsFactors = FALSE)


### 2. PCoA analysis using vegan
library(vegan)

# Calculate Bray–Curtis distance matrix based on OTU abundance
distance <- vegdist(otu, method = "bray")

# Run PCoA (classical multidimensional scaling)
pcoa <- cmdscale(distance,
                 k = (nrow(otu) - 1),
                 eig = TRUE)

# Extract explained variance for the first two axes
pcoa_eig <- (pcoa$eig)[1:2] / sum(pcoa$eig)


### 3. Extract sample coordinates
# Extract PCoA1 and PCoA2 coordinates
sample_site <- data.frame(pcoa$points)[1:2]
sample_site$sample_id <- rownames(sample_site)
names(sample_site)[1:2] <- c("PCoA1", "PCoA2")

# Merge PCoA coordinates with sample metadata
sample_site <- merge(sample_site, group,
                     by.x = "sample_id",
                     by.y = "names",
                     all.x = TRUE)


### 4. Plot PCoA scatter plot using ggplot2
library(ggplot2)
pcoa_plot <- ggplot(sample_site,
                    aes(PCoA1, PCoA2, group = genotype)) +
  theme(
    panel.grid = element_blank(),
    panel.background = element_rect(color = "black",
                                    fill = "transparent"),
    legend.key = element_rect(fill = "transparent")
  ) +
  geom_vline(xintercept = 0, color = "gray", size = 0.3) +
  geom_hline(yintercept = 0, color = "gray", size = 0.3) +
  geom_point(aes(color = genotype, shape = genotype),
             size = 3, alpha = 0.8) +
  scale_shape_manual(values = c(16,16,16,16,16,16,16)) +
  scale_color_manual(values = c("#FF6347","#4682B4","#FFA54F",
                                "#9370DB","#006400","#3CB371","#CDBE70")) +
  labs(
    x = paste("PCoA axis 1 (", round(100 * pcoa_eig[1], 2), "%)", sep = ""),
    y = paste("PCoA axis 2 (", round(100 * pcoa_eig[2], 2), "%)", sep = "")
  )

# Display plot
pcoa_plot

# Save plot
ggsave("results/PCoA_plot.png",
       pcoa_plot, width = 6, height = 5)


### 5. Permutational ANOVA (Adonis test)

# Test significance of group separation using Bray–Curtis distance
adonis_result_otu <- adonis(otu ~ genotype,
                            data = group,
                            permutations = 999,
                            distance = "bray")

# Display results
adonis_result_otu
