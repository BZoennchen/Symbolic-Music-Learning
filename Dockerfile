# Use an official Miniconda image as a parent image
FROM continuumio/miniconda3

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages and Python dependencies using Conda
RUN conda env create -f environment.yml

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Activate the Conda environment
SHELL ["conda", "run", "-n", "sym-learn", "/bin/bash", "-c"]

# The code to run when container is started
ENTRYPOINT ["conda", "run", "-n", "sym-learn", "python", "app.py"]