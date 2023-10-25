action({
    author: 'featherice',
    name: 'emotion change',
    ui: (form) => ({
        info: form.markdown({
            markdown: "This action can change emotion of a character. Add a portrait image and select an emotion to start.",
        }),
        tip: form.markdown({
            markdown: 'For more info feel free to check tooltips.',
            label: ''
        }),
        main: form.group({
            label: "",
            className: ' p-2 bg-blue-800 shadow rounded-xl animate-pulse',
            items: () => ({

                image: form.image({
                }),
                emotion: form.selectOne({
                    choices: [
                        { type: 'happy' },
                        { type: 'sad' },
                        { type: 'angry' },
                        { type: 'disgusted' },
                        { type: 'embarassed' },
                        { type: 'surprised' },
                    ],
                }),
            }),
        }),
        config: form.group({
            items: () => ({
                model: form.group({
                    layout: "H",
                    items: () => ({
                        name: form.enum({
                            enumName: 'Enum_CheckpointLoader_ckpt_name',
                        }),
                        clipSkip: form.int({
                            label: 'Clip Skip',
                            tooltip: 'same as ClipSetLastLayer; you can use both positive and negative values',
                            default: -2,
                            theme: 'input',
                        }),
                        vae: form.enumOpt({
                            enumName: 'Enum_VAELoader_vae_name',
                        }),
                    }),
                }),
                sampler: form.group({
                    layout: "H",
                    items: () => ({
                        name: form.enum({
                            tooltip: 'ddim is recommended to keep character resemblance',
                            enumName: 'Enum_KSampler_sampler_name',
                            default: 'ddim'
                        }),
                        scheduler: form.enum({
                            enumName: 'Enum_KSampler_scheduler',
                            default: 'normal'
                        }),
                    }),
                }),
                denoise: form.group({
                    layout: "H",
                    items: () => ({
                        amount: form.float({
                            label: '%',
                            tooltip: 'Use denoise levels of 0.25-0.35. You can turn on precision mode for better results, with it 0.5 denoise is recommended.',
                            min: 0,
                            max: 1,
                            default: 0.5,
                            step: 0.01,
                            theme: "input"
                        }),
                        steps: form.int({
                            min: 0,
                            max: 100,
                            default: 25,
                            theme: "input"
                        }),
                        cfg: form.float({
                            min: 0,
                            max: 20,
                            default: 8,
                            step: 0.5,
                            theme: "input"
                        }),
                    }),
                }),
                seed: form.seed({
                    default: 0,
                    defaultMode: 'fixed',
                }),
                batch: form.int({
                    min: 1,
                    max: 8,
                    default: 1,
                    theme: "input"
                }),
            }),
        }),
        mask: form.boolean({
            label: "Precise mode",
            tooltip: "Use mask for better control. WARNING: requires controlnet preprocessors and impact pack extra nodes installed in your comfyui!"
        }),
    }),

    run: async (flow, form) => {
        const graph = flow.nodes
        const image = graph.LoadImage({ image: await flow.loadImageAnswerAsEnum(form.main.image) })
        const ckpt = graph.CheckpointLoaderSimple({ ckpt_name: form.config.model.name })
        const vae = form.config.model.vae
            ? graph.VAELoader({ vae_name: form.config.model.vae })
            : ckpt

        const emotion = (() => {
            if (form.main.emotion.type === 'happy') return '(happy_face:1.2)'
            if (form.main.emotion.type === 'angry') return '(angry_face:1.0)'
            if (form.main.emotion.type === 'sad') return '(sad_face:1.1)'
            if (form.main.emotion.type === 'disgusted') return '(disgusted_face:1.2)'
            if (form.main.emotion.type === 'embarassed') return '(embarrassed_face:1.2)'
            if (form.main.emotion.type === 'surprised') return '(surprised_face:1.2)'
        })()
        const positive = graph.CLIPTextEncode({ clip: ckpt, text: '(best_quality, absurdres, highres), ' + emotion })
        const negative = graph.CLIPTextEncode({ clip: ckpt, text: '(worst_quality, low_quality, normal_quality, extra_brows, multiple_brows:1.1)' })

        let latent: HasSingle_LATENT
        if (form.mask) {
            latent = graph.VAEEncode({ vae: vae, pixels: image })
            const facePreprocessor = graph.MediaPipe$7FaceMeshPreprocessor({ image: image, max_faces: 1, min_confidence: 0.6 })
            const faceSEGS = graph.MediaPipeFaceMeshToSEGS({ image: facePreprocessor, face: true })
            const faceMask = graph.SegsToCombinedMask({ segs: faceSEGS })
            latent = graph.SetLatentNoiseMask({ samples: latent, mask: faceMask })
        } else {
            latent = graph.VAEEncode({ vae: vae, pixels: image })
        }
        latent = graph.RepeatLatentBatch({ samples: latent, amount: form.config.batch })
        const images = graph.VAEDecode({
            vae: ckpt,
            samples: graph.KSampler({
                model: ckpt,
                sampler_name: form.config.sampler.name,
                scheduler: form.config.sampler.scheduler,
                seed: form.config.seed,
                steps: form.config.denoise.steps,
                cfg: form.config.denoise.cfg,
                denoise: form.config.denoise.amount,
                positive: positive,
                negative: negative,
                latent_image: latent
            }),
        })
        graph.PreviewImage({ images: images })

        // execute
        await flow.PROMPT()
    }
})
