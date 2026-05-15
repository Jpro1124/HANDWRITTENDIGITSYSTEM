document.addEventListener("DOMContentLoaded", () => {
    const tabButtons = document.querySelectorAll("[data-tab-target]");
    const panels = document.querySelectorAll(".mode-panel");
    const uploadBox = document.querySelector(".upload-box");
    const fileInput = document.querySelector("#digit-image");
    const previewBox = document.querySelector("#upload-preview-box");
    const previewImage = document.querySelector("#upload-preview");
    const modelSelectors = document.querySelectorAll("select[name='model_id']");
    const canvas = document.querySelector("#digit-canvas");
    const canvasShell = document.querySelector(".canvas-shell");
    const drawForm = document.querySelector("#draw-form");
    const drawImageInput = document.querySelector("#draw-image");
    const clearButton = document.querySelector("#clear-canvas");
    const brushSize = document.querySelector("#brush-size");
    const feedbackIncorrectButton = document.querySelector(".feedback-incorrect");
    const feedbackForm = document.querySelector(".feedback-form");
    let lastSparkTime = 0;

    tabButtons.forEach((button) => {
        button.addEventListener("click", () => {
            const targetId = button.dataset.tabTarget;

            tabButtons.forEach((tab) => tab.classList.remove("active"));
            panels.forEach((panel) => panel.classList.add("hidden"));

            button.classList.add("active");
            document.querySelector(`#${targetId}`).classList.remove("hidden");
        });
    });

    modelSelectors.forEach((selector) => {
        selector.addEventListener("change", () => {
            modelSelectors.forEach((otherSelector) => {
                otherSelector.value = selector.value;
            });
        });
    });

    const showPreview = (file) => {
        if (!file || !file.type.startsWith("image/")) {
            return;
        }

        previewImage.src = URL.createObjectURL(file);
        previewBox.classList.remove("hidden");
        uploadBox.classList.add("has-file");
    };

    const setFileInput = (file) => {
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        fileInput.files = dataTransfer.files;
        showPreview(file);
    };

    if (fileInput) {
        fileInput.addEventListener("change", () => {
            showPreview(fileInput.files[0]);
        });
    }

    if (uploadBox) {
        ["dragenter", "dragover"].forEach((eventName) => {
            uploadBox.addEventListener(eventName, (event) => {
                event.preventDefault();
                uploadBox.classList.add("drag-over");
            });
        });

        ["dragleave", "drop"].forEach((eventName) => {
            uploadBox.addEventListener(eventName, (event) => {
                event.preventDefault();
                uploadBox.classList.remove("drag-over");
            });
        });

        uploadBox.addEventListener("drop", (event) => {
            const file = event.dataTransfer.files[0];

            if (file) {
                setFileInput(file);
            }
        });
    }

    document.addEventListener("paste", (event) => {
        if (!event.clipboardData) {
            return;
        }

        const pastedFile = Array.from(event.clipboardData.files).find((file) => {
            return file.type.startsWith("image/");
        });

        if (pastedFile && !document.querySelector("#upload-panel").classList.contains("hidden")) {
            setFileInput(pastedFile);
        }
    });

    if (feedbackIncorrectButton && feedbackForm) {
        feedbackIncorrectButton.addEventListener("click", () => {
            feedbackForm.classList.remove("hidden");
        });
    }

    if (!canvas) {
        return;
    }

    const context = canvas.getContext("2d");
    let isDrawing = false;
    let hasDrawing = false;

    const resetCanvas = () => {
        context.fillStyle = "#000000";
        context.fillRect(0, 0, canvas.width, canvas.height);
        hasDrawing = false;
        canvasShell.classList.remove("drawing");
    };

    const getCanvasPoint = (event) => {
        const rect = canvas.getBoundingClientRect();
        const clientX = event.touches ? event.touches[0].clientX : event.clientX;
        const clientY = event.touches ? event.touches[0].clientY : event.clientY;

        return {
            x: (clientX - rect.left) * (canvas.width / rect.width),
            y: (clientY - rect.top) * (canvas.height / rect.height)
        };
    };

    const startDrawing = (event) => {
        event.preventDefault();
        isDrawing = true;

        const point = getCanvasPoint(event);
        context.beginPath();
        context.moveTo(point.x, point.y);
        canvasShell.classList.add("drawing");
    };

    const addSpark = (event) => {
        const now = Date.now();

        if (now - lastSparkTime < 90) {
            return;
        }

        lastSparkTime = now;

        const clientX = event.touches ? event.touches[0].clientX : event.clientX;
        const clientY = event.touches ? event.touches[0].clientY : event.clientY;
        const spark = document.createElement("span");

        spark.className = "spark";
        spark.style.left = `${clientX}px`;
        spark.style.top = `${clientY}px`;
        document.body.appendChild(spark);

        window.setTimeout(() => {
            spark.remove();
        }, 560);
    };

    const draw = (event) => {
        if (!isDrawing) {
            return;
        }

        event.preventDefault();

        const point = getCanvasPoint(event);
        context.lineWidth = Number(brushSize.value);
        context.lineCap = "round";
        context.lineJoin = "round";
        context.strokeStyle = "#ffffff";
        context.lineTo(point.x, point.y);
        context.stroke();
        hasDrawing = true;
        addSpark(event);
    };

    const stopDrawing = () => {
        isDrawing = false;
        context.closePath();
        canvasShell.classList.remove("drawing");
    };

    canvas.addEventListener("mousedown", startDrawing);
    canvas.addEventListener("mousemove", draw);
    canvas.addEventListener("mouseup", stopDrawing);
    canvas.addEventListener("mouseleave", stopDrawing);
    canvas.addEventListener("touchstart", startDrawing);
    canvas.addEventListener("touchmove", draw);
    canvas.addEventListener("touchend", stopDrawing);

    clearButton.addEventListener("click", resetCanvas);

    drawForm.addEventListener("submit", (event) => {
        if (!hasDrawing) {
            event.preventDefault();
            alert("Please draw a digit before predicting.");
            return;
        }

        drawImageInput.value = canvas.toDataURL("image/png");
    });

    resetCanvas();
});
