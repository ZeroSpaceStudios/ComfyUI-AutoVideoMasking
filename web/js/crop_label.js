import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "SAMhera.CropLabel",
    async nodeCreated(node) {
        if (node.comfyClass !== "SAMheraCropByBox") return;

        // Add a read-only text widget to display the label
        const labelWidget = node.addWidget("text", "crop_label", "", () => {}, {
            multiline: false,
            disabled: true,
        });
        labelWidget.inputEl && (labelWidget.inputEl.readOnly = true);

        const origOnExecuted = node.onExecuted?.bind(node);
        node.onExecuted = function(output) {
            origOnExecuted?.(output);
            if (output?.text?.length) {
                labelWidget.value = output.text[0];
            }
        };
    },
});
