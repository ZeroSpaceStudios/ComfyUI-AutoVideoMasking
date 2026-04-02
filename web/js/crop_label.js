import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "AVM.CropLabel",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "AVMCropByBox") return;

        const origOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (output) {
            origOnExecuted?.call(this, output);
            const text = output?.text?.[0];
            if (!text) return;

            let w = this.widgets?.find(w => w.name === "_crop_label");
            if (!w) {
                w = this.addWidget("string", "_crop_label", "", () => {}, {
                    multiline: false,
                    serialize: false,
                });
                this.setSize(this.computeSize());
            }
            w.value = text;
            this.setDirtyCanvas(true, true);
        };
    },
});
