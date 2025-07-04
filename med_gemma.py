import nibabel as nib
import numpy as np
from PIL import Image
from transformers import pipeline
import torch
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import math
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import time

# Load environment variables
load_dotenv()

# Verify environment variables
if not os.environ.get("HF_TOKEN"):
    print("Error: HF_TOKEN not found in environment variables.")
    print("Ensure .env file contains HF_TOKEN or set it manually.")
    exit(1)

# Define function to load and preprocess NIfTI slices
def load_and_preprocess_nii_slices(nii_path, axis=2, subsample_step=None):
    """
    Load a NIfTI file and extract 2D slices for plotting and VLM.
    Args:
        nii_path (str): Path to the NIfTI (.nii or .nii.gz) file.
        axis (int): Axis for slicing (default: 2, axial).
        subsample_step (int, optional): Step size for subsampling slices for VLM.
    Returns:
        list[np.ndarray], list[PIL.Image], list[int], tuple: Plot slices, VLM images, slice indices, NIfTI shape.
    """
    try:
        nii_img = nib.load(nii_path)
        image = nii_img.get_fdata()
        nii_shape = image.shape
    except Exception as e:
        raise ValueError(f"Failed to load NIfTI file: {e}")
    
    if image.ndim != 3:
        raise ValueError("Input NIfTI image must be 3D.")
    
    slice_count = image.shape[axis]
    slice_indices = list(range(slice_count))
    if subsample_step:
        vlm_slice_indices = slice_indices[::subsample_step]
    else:
        vlm_slice_indices = slice_indices
    
    plot_slices = []
    vlm_images = []
    for idx in slice_indices:
        if axis == 2:
            slice_data = image[..., idx]
        elif axis == 1:
            slice_data = image[:, idx, :]
        elif axis == 0:
            slice_data = image[idx, :, :]
        else:
            raise ValueError("Axis must be 0, 1, or 2.")
        
        # Normalize to [0, 255] and convert to uint8
        slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min() + 1e-8)
        slice_data_uint8 = (slice_data * 255).astype(np.uint8)
        
        # Store for plotting
        plot_slices.append(slice_data_uint8)
        
        # Convert to PIL Image for VLM if in vlm_slice_indices
        if idx in vlm_slice_indices:
            pil_image = Image.fromarray(slice_data_uint8)
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            vlm_images.append(pil_image)
    
    return plot_slices, vlm_images, vlm_slice_indices, nii_shape

# Initialize the VLM pipeline for slice analysis
try:
    pipe = pipeline(
        "image-text-to-text",
        model="google/medgemma-4b-it",
        torch_dtype=torch.bfloat16,
        device="cuda",  # Change to "cpu" if no GPU
        token=os.environ["HF_TOKEN"]
    )
except Exception as e:
    print(f"Error loading VLM pipeline: {e}")
    print("Ensure transformers is up-to-date: !pip install --upgrade transformers")
    exit(1)

# Initialize Ollama LLM for summary report
try:
    ollama_llm = ChatOllama(
        model="gemma3:27b",
        temperature=0.1,

    )
except Exception as e:
    print(f"Error initializing Ollama LLM: {e}")
    print("Ensure langchain-ollama is installed: !pip install langchain-ollama")
    print("Ensure Ollama server is running with the specified model.")
    exit(1)

#  Load and preprocess all slices from the NIfTI file
nii_path = "data/BraTS-MET-00008-000-seg.nii"
try:
    plot_slices, vlm_images, slice_indices, nii_shape = load_and_preprocess_nii_slices(nii_path, axis=2, subsample_step=None)
except Exception as e:
    print(f"Error loading NIfTI file: {e}")
    exit(1)

# Print NIfTI shape and slice info
print(f"NIfTI File Shape: {nii_shape}")
print(f"Total Slices (axial): {nii_shape[2]}")
print(f"Number of Slices for Plotting: {len(plot_slices)}")
print(f"Number of Slices for VLM Analysis: {len(vlm_images)}")
print(f"VLM Slice Indices: {slice_indices}")

# Plot all slices in multiple figures
slices_per_figure = 20
num_slices = len(plot_slices)
num_figures = math.ceil(num_slices / slices_per_figure)

for fig_idx in range(num_figures):
    start_idx = fig_idx * slices_per_figure
    end_idx = min((fig_idx + 1) * slices_per_figure, num_slices)
    current_slices = plot_slices[start_idx:end_idx]
    current_indices = list(range(start_idx, end_idx))
    
    rows = 4
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()
    
    for i, (slice_data, slice_idx) in enumerate(zip(current_slices, current_indices)):
        if i >= len(axes):
            break
        axes[i].imshow(slice_data, cmap="gray")
        axes[i].set_title(f"Slice {slice_idx}")
        axes[i].axis("off")
    
    for i in range(len(current_slices), len(axes)):
        axes[i].axis("off")
    
    plt.suptitle(f"T2-Weighted MRI Slices (Axial) - Figure {fig_idx + 1}/{num_figures}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# Run VLM on all slices and collect descriptions with timing
descriptions = []
slice_times = []
output_file = "mri_radiology_report_exp.txt"

# Open file to save reports
with open(output_file, "w") as f:
    f.write("Detailed MRI Scan Analysis:\n\n")

for i, (image, slice_idx) in enumerate(zip(vlm_images, slice_indices)):
    start_time = time.time()
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an expert radiologist tasked with generating a structured radiology report for an MRI brain scan. Provide concise, accurate, and professional responses in the specified format."}]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Generate a structured radiology report for slice {slice_idx} of an axial T2-weighted MRI brain scan. Classify any abnormalities as one of: Tumor, Multiple Sclerosis, Stroke, Tuberculosis, Others, or No Anomaly. Use the following format:\n\n**Radiology Report - Slice {slice_idx}**\n**Anatomical Structures**: List all visible anatomical structures.\n**Findings**: If an abnormality is present, classify it into one of (Tumor, Multiple Sclerosis, Stroke, Tuberculosis, Others) and describe its approximate size in pixels, precise location, signal intensity, presence of mass effect, necrosis, or enhancement patterns, and impact on adjacent structures. If no abnormality is present, state 'No Anomaly detected.'\n**Impression**: Summarize the primary findings, including the classified abnormality and its potential clinical implications. If no abnormality is detected, state 'No Anomaly detected, normal slice with no clinical concerns.'\n\nEnsure the report is concise, professional, and avoids speculative language."
                },
                {"type": "image", "image": image}
            ]
        }
    ]
    try:
        output = pipe(text=messages, max_new_tokens=500)
        description = output[0]["generated_text"][-1]["content"]
        descriptions.append(description)
    except Exception as e:
        print(f"Error running pipeline for slice {slice_idx}: {e}")
        description = f"**Radiology Report - Slice {slice_idx}**\n**Error**: Failed to process slice."
        descriptions.append(description)
    end_time = time.time()
    slice_times.append(end_time - start_time)
    
    # Save per-slice report to file
    with open(output_file, "a") as f:
        f.write(f"Time taken for Slice {slice_idx} report: {slice_times[-1]:.2f} seconds\n")
        f.write(f"{description}\n")
        f.write("\n" + "-"*50 + "\n")

# Print individual slice descriptions with timing
print("\nDetailed MRI Scan Analysis:")
for idx, (desc, t) in enumerate(zip(descriptions, slice_times)):
    print(f"Time taken for Slice {slice_indices[idx]} report: {t:.2f} seconds")
    print(desc)
    print("\n" + "-"*50 + "\n")

# Generate overall radiology report using LangChain and Ollama with timing
summary_prompt = PromptTemplate.from_template(
    """
    You are an expert radiologist tasked with generating a comprehensive radiology report summarizing findings from multiple axial T2-weighted MRI brain scan slices. Below are the individual slice reports:

    {slice_reports}

    Generate a structured overall radiology report in the following format:

    **Overall Radiology Report - T2-Weighted MRI Brain Scan**
    **Summary of Findings**:
    - List all anatomical regions examined across all slices.
    - Detail significant abnormalities, classified as Tumor, Multiple Sclerosis, Stroke, Tuberculosis, Others, or No Anomaly. For each abnormality, include approximate size range in pixels, precise locations, signal intensity, presence of mass effect, necrosis, or enhancement patterns, and impact on adjacent structures. If no abnormalities are detected, state 'No Anomaly detected across all slices.'
    **Impression**:
    - Provide a detailed summary of the primary findings, including the classified abnormalities and their potential clinical implications.
    - Include recommendations for further evaluation or state 'Routine follow-up imaging recommended' if no abnormalities are detected.
    **Note**: Ensure the report is concise, professional, and avoids speculative language. Synthesize information from all slices for a cohesive overview.
    """
)

# Create LangChain pipeline for summary
summary_chain = summary_prompt | ollama_llm | StrOutputParser()

# Combine slice reports into a single string
slice_reports_text = "\n\n".join(descriptions)

# Generate and print overall report with timing
start_time = time.time()
try:
    overall_report = summary_chain.invoke({"slice_reports": slice_reports_text})
    end_time = time.time()
    overall_time = end_time - start_time
except Exception as e:
    print(f"Error generating overall report: {e}")
    overall_report = "Error: Failed to generate overall report."
    overall_time = 0

# Calculate total time
total_time = sum(slice_times) + overall_time

# Save overall report to file
with open(output_file, "a") as f:
    f.write(f"\nTime taken for Overall Report generation: {overall_time:.2f} seconds\n")
    f.write("\nOverall MRI Scan Analysis:\n")
    f.write(overall_report)

# Print total time and overall report
print(f"\nTotal time taken for all reports (slices + overall): {total_time:.2f} seconds")
print(f"\nTime taken for Overall Report generation: {overall_time:.2f} seconds")
print("\nOverall MRI Scan Analysis:")
print(overall_report)