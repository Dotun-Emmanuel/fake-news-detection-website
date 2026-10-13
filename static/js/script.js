const newsInput = document.getElementById("newsText");
const predictButton = document.getElementById("predictButton");
const clearButton = document.getElementById("clearButton");
const resultWordCount = document.getElementById("resultWordCount");
const checkAnotherButton = document.getElementById("checkAnotherButton");

const emptyState = document.getElementById("emptyState");
const loadingState = document.getElementById("loadingState");
const resultState = document.getElementById("resultState");
const errorState = document.getElementById("errorState");

const resultLabel = document.getElementById("resultLabel");
const resultMessage = document.getElementById("resultMessage");
const errorMessage = document.getElementById("errorMessage");

function showState(state) {
    [emptyState, loadingState, resultState, errorState].forEach((element) => {
        element.classList.add("hidden");
    });
    state.classList.remove("hidden");
}

predictButton.addEventListener("click", async () => {
    const article = newsInput.value.trim();

    if (!article) {
        errorMessage.textContent = "Please enter a news article.";
        showState(errorState);
        return;
    }

    predictButton.disabled = true;
    showState(loadingState);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ news_text: article })
        });

        const data = await response.json();

        if (!response.ok) {
            errorMessage.textContent = data.error || "Prediction failed.";
            showState(errorState);
            return;
        }

        const isReal = data.prediction === "Real News";

        resultLabel.textContent = data.prediction;
        resultMessage.textContent = isReal
            ? "The article matches patterns of real news in the training data."
            : "The article matches patterns of fake news in the training data.";

        resultWordCount.textContent =
            article.trim().split(/\s+/).length + " words";

        resultState.classList.remove("real", "fake");
        resultState.classList.add(isReal ? "real" : "fake");

        showState(resultState);
    } catch (error) {
        errorMessage.textContent = "Could not connect to the server.";
        showState(errorState);
    } finally {
        predictButton.disabled = false;
    }
});

clearButton.addEventListener("click", () => {
    newsInput.value = "";
    showState(emptyState);
});

checkAnotherButton.addEventListener("click", () => {
    newsInput.value = "";
    showState(emptyState);
    newsInput.focus();
}

);