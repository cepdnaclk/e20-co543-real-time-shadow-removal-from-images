document.addEventListener("DOMContentLoaded", () => {
    const imageInput = document.getElementById("image");
    const beforeImageContainer = document.getElementById("before-image");
    const afterImageContainer = document.getElementById("after-image");
    const downloadContainer = document.getElementById("download-container");
    const downloadButton = document.getElementById("download-button");
    const form = document.getElementById("image-form");

    // Handle image upload preview
    if (imageInput) {
        imageInput.addEventListener("change", (event) => {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    if (beforeImageContainer) {
                        beforeImageContainer.src = e.target.result; // Display the uploaded file
                    }
                };
                reader.readAsDataURL(file); // Convert the file to a Data URL
            }
        });
    }

    // Handle form submission for processing
    if (form) {
        form.addEventListener("submit", (event) => {
            event.preventDefault();

            const loadingMessage = document.createElement("div");
            loadingMessage.className = "alert alert-info text-center";
            loadingMessage.textContent = "Processing image, please wait...";
            form.insertAdjacentElement("beforebegin", loadingMessage);

            const formData = new FormData(form);

            fetch("/process", {
                method: "POST",
                body: formData,
            })
                .then((response) => response.json())
                .then((data) => {
                    loadingMessage.remove();
                    if (data.processed_image && data.uploaded_image) {
                        // Update image placeholders
                        if (beforeImageContainer) {
                            beforeImageContainer.src = data.uploaded_image;
                        }
                        if (afterImageContainer) {
                            afterImageContainer.src = data.processed_image;
                        }

                        // Show download button
                        if (downloadButton) {
                            downloadButton.href = data.processed_image;
                            downloadContainer.style.display = "block";
                        }
                    } else {
                        const errorMessage = document.createElement("div");
                        errorMessage.className = "alert alert-danger text-center";
                        errorMessage.textContent = data.error || "An error occurred while processing the image.";
                        form.insertAdjacentElement("beforebegin", errorMessage);
                    }
                })
                .catch((error) => {
                    loadingMessage.remove();
                    const errorMessage = document.createElement("div");
                    errorMessage.className = "alert alert-danger text-center";
                    errorMessage.textContent = "An error occurred while processing the image.";
                    form.insertAdjacentElement("beforebegin", errorMessage);
                });
        });
    }
});
