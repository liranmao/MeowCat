# ================== RCTD deconvolution (Seurat v5 reference) ==================
#
# Usage (called by meowcat rctd):
#   Rscript RCTD_deconvolution.R \
#     --base_dir /path/to/data \
#     --sample_pattern "VIS*" \
#     --reference_rds /path/to/reference.rds \
#     --cell_type_column MainType \
#     --max_cores 5 \
#     --doublet_mode full \
#     --min_umi 10 \
#     [--group_column Group_ID] \
#     [--groups LUAD,MIA,AIS,AAH,Normal]
#
# All arguments are passed from config.yaml via meowcat/pipeline.py.
# ==============================================================================
library(spacexr)
library(Matrix)
library(Seurat)

set.seed(42)

# ---------- argument parsing (no external package needed) ----------
parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  # Remove --no-save if present (legacy compat)
  args <- args[args != "--no-save"]

  opts <- list(
    base_dir         = NULL,
    sample_pattern   = "*",
    reference_rds    = NULL,
    cell_type_column = "MainType",
    group_column     = "",
    groups           = character(0),
    max_cores        = 5L,
    doublet_mode     = "full",
    min_umi          = 10L
  )

  i <- 1
  while (i <= length(args)) {
    key <- sub("^--", "", args[i])
    if (i + 1 <= length(args)) {
      val <- args[i + 1]
    } else {
      stop("Missing value for argument: ", args[i])
    }
    if (key == "max_cores" || key == "min_umi") {
      opts[[key]] <- as.integer(val)
    } else if (key == "groups") {
      opts[[key]] <- strsplit(val, ",")[[1]]
    } else {
      opts[[key]] <- val
    }
    i <- i + 2
  }

  if (is.null(opts$base_dir))       stop("--base_dir is required")
  if (is.null(opts$reference_rds))  stop("--reference_rds is required")

  opts
}

opts <- parse_args()

# ---------- helpers ----------
collapse_duplicate_genes <- function(mat) {
  if (anyDuplicated(rownames(mat))) {
    mat <- as(rowsum(as.matrix(mat), group = rownames(mat)), "dgCMatrix")
  }
  mat
}

read_positions_visium <- function(spatial_dir) {
  tp_csv  <- file.path(spatial_dir, "tissue_positions.csv")
  tp_list <- file.path(spatial_dir, "tissue_positions_list.csv")
  f <- if (file.exists(tp_csv)) tp_csv else tp_list
  stopifnot(file.exists(f))
  first_line <- readLines(f, n = 1)
  has_header <- grepl("barcode", first_line, ignore.case = TRUE)
  df <- read.csv(f, header = has_header, stringsAsFactors = FALSE)
  if (!has_header) {
    colnames(df) <- c("barcode","in_tissue","array_row","array_col",
                      "pxl_row_in_fullres","pxl_col_in_fullres")
  }
  df
}

# Build SpatialRNA from a Visium sample directory that has 10x outputs
read_visium_as_spatialRNA <- function(sample_dir) {
  mtx_dir   <- file.path(sample_dir, "filtered_feature_bc_matrix")
  mtx_path  <- file.path(mtx_dir, "matrix.mtx.gz")
  feats_path<- file.path(mtx_dir, "features.tsv.gz")
  barc_path <- file.path(mtx_dir, "barcodes.tsv.gz")
  stopifnot(file.exists(mtx_path), file.exists(feats_path), file.exists(barc_path))

  M        <- Matrix::readMM(mtx_path)
  feat_df  <- read.delim(feats_path, header = FALSE, stringsAsFactors = FALSE)
  barcodes <- read.delim(barc_path,  header = FALSE, stringsAsFactors = FALSE)[[1]]

  # Keep only Gene Expression features; use gene symbols (V2)
  if (ncol(feat_df) >= 3) {
    keep <- feat_df[[3]] == "Gene Expression"
    M       <- M[keep, , drop = FALSE]
    feat_df <- feat_df[keep, , drop = FALSE]
  }
  gene_symbols <- if (ncol(feat_df) >= 2 && any(nzchar(feat_df[[2]]))) feat_df[[2]] else feat_df[[1]]
  rownames(M) <- gene_symbols
  colnames(M) <- barcodes

  counts <- as(M, "dgCMatrix")
  counts <- collapse_duplicate_genes(counts)

  # coordinates, keep in-tissue
  pos <- read_positions_visium(file.path(sample_dir, "spatial"))
  pos <- subset(pos, in_tissue == 1)
  coords <- pos[, c("pxl_col_in_fullres","pxl_row_in_fullres")]
  rownames(coords) <- pos$barcode
  colnames(coords) <- c("x","y")

  # align barcodes
  bc <- intersect(colnames(counts), rownames(coords))
  counts <- counts[, bc, drop = FALSE]
  coords <- coords[bc, , drop = FALSE]

  SpatialRNA(coords, counts)
}

# simple plotting wrapper with robust cutoff fallback
plot_weights <- function(cell_type_names, puck, outfile_pdf, weights) {
  pdf(outfile_pdf)
  for (cell_type in cell_type_names) {
    cutoff <- tryCatch(spacexr:::UMI_cutoff(puck@nUMI),
                       error = function(e) 0)
    my_cond <- weights[, cell_type] > cutoff
    plot_var <- weights[, cell_type]; names(plot_var) <- rownames(weights)
    if (sum(my_cond) > 0) {
      p <- plot_puck_wrapper(
        puck, plot_var, NULL,
        minUMI = 100, maxUMI = 200000,
        min_val = 0, max_val = 1,
        title = cell_type, my_cond = my_cond
      )
      print(p)
    }
  }
  dev.off()
}

# Helper to build a reference for one group
build_reference_for_group <- function(singlecell_all, group_name,
                                      group_column, cell_type_column,
                                      min_UMI = 10) {
  message("Building reference for group: ", group_name)

  sc_sub <- subset(singlecell_all, cells = colnames(singlecell_all)[
    singlecell_all@meta.data[[group_column]] == group_name
  ])
  if (ncol(sc_sub) == 0) {
    stop("No cells found in singlecell object for group: ", group_name)
  }

  DefaultAssay(sc_sub) <- "RNA"

  # Seurat v5: pull counts from the RNA assay "counts" layer
  counts_sc <- LayerData(sc_sub$RNA, layer = "counts")
  counts_sc <- as(counts_sc, "dgCMatrix")

  # Labels
  labels <- sc_sub@meta.data[colnames(counts_sc), cell_type_column]
  names(labels) <- colnames(counts_sc)

  counts_sc <- collapse_duplicate_genes(counts_sc)

  reference_major <- Reference(counts_sc, labels, min_UMI = min_UMI)
  message("Reference for ", group_name, " built with ",
          ncol(counts_sc), " cells and ",
          nrow(counts_sc), " genes across ",
          length(levels(labels)), " cell types.")

  return(reference_major)
}

# Helper to build a single reference from all cells (no group subsetting)
build_reference_all <- function(singlecell_all, cell_type_column, min_UMI = 10) {
  message("Building reference from all cells (no group subsetting)")

  DefaultAssay(singlecell_all) <- "RNA"

  counts_sc <- LayerData(singlecell_all$RNA, layer = "counts")
  counts_sc <- as(counts_sc, "dgCMatrix")

  labels <- singlecell_all@meta.data[colnames(counts_sc), cell_type_column]
  names(labels) <- colnames(counts_sc)

  counts_sc <- collapse_duplicate_genes(counts_sc)

  reference_major <- Reference(counts_sc, labels, min_UMI = min_UMI)
  message("Reference built with ",
          ncol(counts_sc), " cells and ",
          nrow(counts_sc), " genes across ",
          length(levels(labels)), " cell types.")

  return(reference_major)
}

# Helper: infer group from sample_id like "P1_LUAD", "P10_MIA", "P4_Normal"
infer_group_from_sample_id <- function(sample_id, groups) {
  hits <- groups[sapply(groups, function(g) grepl(g, sample_id, fixed = TRUE))]
  if (length(hits) == 1) return(hits)
  if (length(hits) == 0) {
    warning("Could not infer group from sample_id: ", sample_id)
  } else {
    warning("Multiple groups matched for sample_id: ", sample_id,
            " -> ", paste(hits, collapse = ", "))
  }
  return(NA_character_)
}

# ---------- discover samples ----------
sample_dirs <- Sys.glob(file.path(opts$base_dir, opts$sample_pattern))
sample_dirs <- sample_dirs[file.info(sample_dirs)$isdir]
# Keep only samples that have Visium inputs (filtered_feature_bc_matrix/ + spatial/)
sample_dirs <- sample_dirs[sapply(sample_dirs, function(d) {
  dir.exists(file.path(d, "filtered_feature_bc_matrix")) &&
    dir.exists(file.path(d, "spatial"))
})]

if (length(sample_dirs) == 0) {
  message("No Visium samples found under ", opts$base_dir,
          " matching pattern '", opts$sample_pattern, "'")
  quit(status = 0)
}
message("Found ", length(sample_dirs), " Visium sample(s): ",
        paste(basename(sample_dirs), collapse = ", "))

# ---------- load single-cell reference ----------
singlecell_all <- readRDS(opts$reference_rds)

stopifnot(opts$cell_type_column %in% colnames(singlecell_all@meta.data))
DefaultAssay(singlecell_all) <- "RNA"

use_groups <- nzchar(opts$group_column) && length(opts$groups) > 0
if (use_groups) {
  stopifnot(opts$group_column %in% colnames(singlecell_all@meta.data))
  message("Group-based subsetting enabled: column='", opts$group_column,
          "', groups=", paste(opts$groups, collapse = ","))
} else {
  message("No group subsetting; using all cells as one reference.")
}

# Lazy cache of references so we build each group only once
reference_cache <- list()

# If no group subsetting, build a single reference up front
if (!use_groups) {
  reference_single <- build_reference_all(singlecell_all, opts$cell_type_column,
                                          min_UMI = opts$min_umi)
}

# ---------- run per-sample ----------
for (sample_dir in sample_dirs) {
  sample_id <- basename(sample_dir)
  message("\n===== RCTD on sample: ", sample_id, " =====")

  if (use_groups) {
    # Infer group from folder name
    sample_group <- infer_group_from_sample_id(sample_id, opts$groups)
    if (is.na(sample_group)) {
      message("Skipping sample ", sample_id, " because group could not be inferred.")
      next
    }
    message("Sample ", sample_id, " assigned to group: ", sample_group)

    # Get or build reference for this group
    if (!sample_group %in% names(reference_cache)) {
      reference_cache[[sample_group]] <- build_reference_for_group(
        singlecell_all, sample_group,
        opts$group_column, opts$cell_type_column,
        min_UMI = opts$min_umi
      )
    }
    reference_major <- reference_cache[[sample_group]]
  } else {
    reference_major <- reference_single
  }

  # Output paths
  out_dir <- file.path(sample_dir, "deconvolution_rctd")
  if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
  csv_out <- file.path(out_dir, "major_prop.csv")
  pdf_out <- file.path(out_dir, "major_prop.pdf")

  if (file.exists(csv_out)) {
    message("Already done: ", csv_out, " \u2014 skipping.")
    next
  }

  # Build SpatialRNA from 10x files
  puck <- read_visium_as_spatialRNA(sample_dir)

  # Fit
  myRCTD <- create.RCTD(puck, reference_major, max_cores = opts$max_cores)
  myRCTD <- run.RCTD(myRCTD, doublet_mode = opts$doublet_mode)

  # Normalize to proportions, save CSV + PDF
  nw <- normalize_weights(myRCTD@results$weights)
  rownames(nw) <- colnames(myRCTD@spatialRNA@counts)
  nw <- as.matrix(nw)

  write.csv(nw, csv_out, quote = FALSE)
  cell_type_names <- myRCTD@cell_type_info$info[[2]]
  plot_weights(cell_type_names, myRCTD@spatialRNA, pdf_out, nw)

  message("Saved: ", csv_out, " and ", pdf_out)

  rm(puck, myRCTD, nw, cell_type_names)
  gc()
}
quit(status = 0)                      
                 
