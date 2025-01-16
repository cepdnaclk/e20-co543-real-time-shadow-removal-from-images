document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const imageInput = document.getElementById("image");
    const beforeImageContainer = document.querySelector(".col-md-6.text-center .img-container img[alt='Uploaded Image']");
    const afterImageContainer = document.querySelector(".col-md-6.text-center .img-container img[alt='Processed Image']");
    const form = document.querySelector("form");

    // Event Listener: Display selected image in the "Before" section
    if (imageInput) {
        imageInput.addEventListener("change", (event) => {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    beforeImageContainer.src = e.target.result;
                };
                reader.readAsDataURL(file);
            }
        });
    }

    // Form Submit Handler: Show loading indicator and send data to backend
    if (form) {
        form.addEventListener("submit", (event) => {
            event.preventDefault(); // Prevent default form submission for demo purposes

            // Show a loading spinner or message
            const loadingMessage = document.createElement("div");
            loadingMessage.className = "alert alert-info text-center";
            loadingMessage.textContent = "Processing image, please wait...";
            form.insertAdjacentElement("beforebegin", loadingMessage);

            const formData = new FormData(form);

            // Send image to backend for processing
            fetch("/process", {
                method: "POST",
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.uploaded_image && data.processed_image) {
                    afterImageContainer.src = data.processed_image;
                    beforeImageContainer.src = data.uploaded_image;
                } else {
                    // Handle error from backend
                    loadingMessage.remove();
                    const errorMessage = document.createElement("div");
                    errorMessage.className = "alert alert-danger text-center";
                    errorMessage.textContent = data.error || "An error occurred while processing the image.";
                    form.insertAdjacentElement("beforebegin", errorMessage);
                }
            })
            .catch(error => {
                loadingMessage.remove();
                const errorMessage = document.createElement("div");
                errorMessage.className = "alert alert-danger text-center";
                errorMessage.textContent = "An error occurred while processing the image.";
                form.insertAdjacentElement("beforebegin", errorMessage);
            });
        });
    }
});
