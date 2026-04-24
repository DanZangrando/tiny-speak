import graphviz

def get_common_node_style():
    return 'fontname="Helvetica", shape=box, style="filled,rounded", fontcolor="white"'

def get_common_edge_style():
    return 'fontname="Helvetica", color="#adb5bd", fontcolor="#adb5bd"'

def get_recognizer_diagram():
    dot = graphviz.Digraph('VisualPathway', comment='VisualPathway Architecture')
    dot.attr(rankdir='TB', bgcolor='#0e1117', fontname='Helvetica', fontcolor='white')
    
    # Global Styles
    dot.attr('node', **{'fontname': 'Helvetica', 'shape': 'box', 'style': 'filled,rounded', 'fontcolor': 'white', 'color': '#303030', 'margin': '0.2'})
    dot.attr('edge', **{'fontname': 'Helvetica', 'color': '#6c757d', 'fontcolor': '#adb5bd', 'penwidth': '1.5'})

    # Input
    dot.node('Input', 'Input Image\n(3 x 64 x 64)\nRGB', fillcolor='#1e88e5', color='#1565c0', shape='note')

    # Backbone
    with dot.subgraph(name='cluster_backbone') as c:
        c.attr(color='#4facfe', fontcolor='#4facfe', style='dashed', penwidth='2')
        
        c.node('V1', 'Block 1\nConv2d(3→64, k=3)\nMaxPool(2x2)', fillcolor='#263238')
        c.node('V2', 'Block 2\nConv2d(64→128, k=3)\nMaxPool(2x2)', fillcolor='#263238')
        c.node('V4', 'Block 3\nConv2d(128→256, k=3)\nMaxPool(2x2)', fillcolor='#263238')
        c.node('IT', 'Block 4\nConv2d(256→512, k=3)\nMaxPool(2x2)', fillcolor='#263238')

    # Decoder
    with dot.subgraph(name='cluster_decoder') as c:
        c.attr(label='Classifier Head', color='#ff0080', fontcolor='#ff0080', style='dashed', penwidth='2')
        
        c.node('AvgPool', 'AdaptiveAvgPool2d\n(1x1)', fillcolor='#424242', shape='ellipse')
        c.node('Flatten', 'Flatten\n(Batch, 512)', fillcolor='#424242', shape='ellipse')
        c.node('Classifier', 'Linear(512 → NumClasses)', fillcolor='#d81b60', shape='box')

    # Output
    dot.node('Output', 'Logits / Probabilities\n(Batch, NumClasses)', fillcolor='#43a047', color='#2e7d32', shape='parallelogram')

    # Edges
    dot.edge('Input', 'V1')
    dot.edge('V1', 'V2')
    dot.edge('V2', 'V4')
    dot.edge('V4', 'IT')
    dot.edge('IT', 'AvgPool')
    dot.edge('AvgPool', 'Flatten')
    dot.edge('Flatten', 'Classifier')
    dot.edge('Classifier', 'Output')

    return dot

def get_listener_diagram():
    dot = graphviz.Digraph('PhonologicalPathway', comment='PhonologicalPathway Architecture')
    dot.attr(rankdir='TB', bgcolor='#0e1117', fontname='Helvetica', fontcolor='white')
    
    dot.attr('node', **{'fontname': 'Helvetica', 'shape': 'box', 'style': 'filled,rounded', 'fontcolor': 'white', 'color': '#303030', 'margin': '0.2'})
    dot.attr('edge', **{'fontname': 'Helvetica', 'color': '#6c757d', 'fontcolor': '#adb5bd', 'penwidth': '1.5'})

    # Input
    dot.node('Input', 'Audio Waveform\n(Batch, Samples)', fillcolor='#1e88e5', color='#1565c0', shape='note')

    # Preprocessing
    with dot.subgraph(name='cluster_preprocessing') as c:
        c.attr(label='Auditory Preprocessing', color='#00f2fe', fontcolor='#00f2fe', style='dashed', penwidth='2')
        c.node('MelSpec', 'Mel Spectrogram\n(80 bands)', fillcolor='#0097a7', shape='component')
        c.node('LogMel', 'Log Scaling', fillcolor='#00838f', shape='ellipse')

    # Feature Extractor
    with dot.subgraph(name='cluster_feature_extractor') as c:
        c.attr(label='Feature Extractor (CNN)', color='#4facfe', fontcolor='#4facfe', style='filled', fillcolor='#1a2327')
        c.node('Conv1', 'Conv1d Block\n(k=5, s=2)', fillcolor='#37474f')
        c.node('Conv2', 'Conv1d Block\n(k=5, s=2)', fillcolor='#37474f')
        c.node('Conv3', 'Conv1d Block\n(k=5, s=2)', fillcolor='#37474f')
        c.node('Proj', 'Linear Projection\n(Channels → Hidden)', fillcolor='#37474f')

    # Positional Encoding
    dot.node('PosEnc', 'Positional Encoding\n(Sinusoidal)', fillcolor='#fdd835', fontcolor='black', shape='diamond')

    # Encoder
    with dot.subgraph(name='cluster_encoder') as c:
        c.attr(label='Context Encoder (Transformer)', color='#ff0080', fontcolor='#ff0080', style='dashed', penwidth='2')
        c.node('Transformer', 'Transformer Encoder\n(2 Layers, 4 Heads)', fillcolor='#d81b60')
        c.node('Pooling', 'Mean Pooling\n(Time Dimension)', fillcolor='#ff6f00', shape='ellipse')

    dot.node('Head', 'Classification Head\nLinear(Hidden → NumClasses)', fillcolor='#d81b60')
    dot.node('Output', 'Logits\n(Batch, NumClasses)', fillcolor='#43a047', color='#2e7d32', shape='parallelogram')

    # Edges
    dot.edge('Input', 'MelSpec')
    dot.edge('MelSpec', 'LogMel')
    dot.edge('LogMel', 'Conv1')
    dot.edge('Conv1', 'Conv2')
    dot.edge('Conv2', 'Conv3')
    dot.edge('Conv3', 'Proj')
    dot.edge('Proj', 'PosEnc')
    dot.edge('PosEnc', 'Transformer', label='+')
    dot.edge('Transformer', 'Pooling')
    dot.edge('Pooling', 'Head')
    dot.edge('Head', 'Output')

    return dot

def get_latent_mapping_diagram(module_name="TinySpeller", stage="G2P"):
    """
    Versión simplificada y técnica del mapeo entre espacios latentes.
    """
    dot = graphviz.Digraph(comment='Latent Mapping Diagram')
    dot.attr(rankdir='LR', bgcolor='#0e1117', fontcolor='white')
    dot.attr('node', style='filled', fontname='Helvetica', color='white', fontcolor='white', shape='box', border='2')
    
    # Estilo de nodos
    visual_color = '#00838f'
    neural_color = '#880e4f'
    auditory_color = '#ab47bc'
    
    if stage == "G2P":
        dot.node('Input', 'Input:\nGrafema (Imagen/ID)', fillcolor='#424242')
        dot.node('Encoder', 'Sensor Visual:\nTinyEyes (Frozen)', fillcolor=visual_color)
        dot.node('LatentV', 'Neural Image (IT):\nEspacio Visual (512-d)', fillcolor=visual_color, shape='parallelogram')
        dot.node('Projector', f'Proyector:\n{module_name}\n(Pointwise MLP)', fillcolor=neural_color, style='filled,bold')
        dot.node('LatentA', 'Neural Image (Phonetic):\nEspacio Auditivo (256-d)', fillcolor=auditory_color, shape='parallelogram')
        dot.node('Decoder', 'Sensor Auditivo:\nTinyEars (Frozen)', fillcolor=auditory_color)
        
        dot.edge('Input', 'Encoder')
        dot.edge('Encoder', 'LatentV')
        dot.edge('LatentV', 'Projector')
        dot.edge('Projector', 'LatentA')
        dot.edge('LatentA', 'Decoder')
        
        # Loss
        dot.node('Loss', 'Perceptual Loss\n(Neural Match)', color='#ff5252', fontcolor='#ff5252', shape='circle')
        dot.edge('LatentA', 'Loss', color='#ff5252', style='dashed')
        dot.edge('Decoder', 'Loss', color='#ff5252', style='dashed', label='Target')

    else:
        dot.node('Input', 'Input:\nSecuencia de Fonemas', fillcolor=auditory_color)
        dot.node('Projector', f'Integrador:\n{module_name}\n(Sequential Bi-LSTM)', fillcolor=neural_color, style='filled,bold')
        dot.node('LatentW', 'Neural Image (Word):\nRepresentación Léxica', fillcolor='#00acc1', shape='parallelogram')
        dot.node('Decoder', 'Sensor Auditivo Words:\nTinyEars (Frozen)', fillcolor='#00acc1')
        
        dot.edge('Input', 'Projector')
        dot.edge('Projector', 'LatentW')
        dot.edge('LatentW', 'Decoder')
        
        # Loss
        dot.node('Loss', 'Perceptual Loss\n(Semantic Match)', color='#ff5252', fontcolor='#ff5252', shape='circle')
        dot.edge('LatentW', 'Loss', color='#ff5252', style='dashed')
        dot.edge('Decoder', 'Loss', color='#ff5252', style='dashed', label='Target')

    return dot

def get_reader_diagram():
    dot = graphviz.Digraph('TinyReader', comment='TinyReader Two-Stage Architecture')
    dot.attr(rankdir='TB', bgcolor='#0e1117', fontname='Helvetica', fontcolor='white')
    
    # Global Styles
    dot.attr('node', **{'fontname': 'Helvetica', 'shape': 'box', 'style': 'filled,rounded', 'fontcolor': 'white', 'color': '#303030', 'margin': '0.2'})
    dot.attr('edge', **{'fontname': 'Helvetica', 'color': '#6c757d', 'fontcolor': '#adb5bd', 'penwidth': '1.5'})

    # ==========================================
    # 1. VISUAL PATHWAY (Input)
    # ==========================================
    with dot.subgraph(name='cluster_visual') as c:
        c.attr(label='Visual Pathway (The "Eye")', color='#4facfe', fontcolor='#4facfe', style='dashed', penwidth='2')
        
        c.node('VisInput', 'Input Image\n(Grapheme)', fillcolor='#1e88e5', shape='note')
        c.node('VisCNN', 'Visual CNN\n(V1 -> IT)', fillcolor='#263238')
        c.node('VisLogits', 'Grapheme Logits\n(Concept Vector)', fillcolor='#d81b60', shape='parallelogram')
        
        c.edge('VisInput', 'VisCNN')
        c.edge('VisCNN', 'VisLogits')

    # ==========================================
    # 2. TINY READER (Two-Stage Generator)
    # ==========================================
    with dot.subgraph(name='cluster_reader') as c:
        c.attr(label='TinyReader (The "Inner Voice")', color='#ff0080', fontcolor='#ff0080', style='dashed', penwidth='2')
        
        # Stage 1: G2P
        with c.subgraph(name='cluster_g2p') as g2p:
            g2p.attr(label='Stage 1: Grapheme-to-Phoneme', color='#ff4081', fontcolor='#ff4081', style='dotted')
            g2p.node('ReaderG2P', 'TinySpeller\n(Pointwise MLP)', fillcolor='#880e4f')
            g2p.node('PhonemeEmbed', 'Phoneme Embeddings\n(Neural Image)', fillcolor='#ab47bc', shape='parallelogram')
            
            g2p.edge('ReaderG2P', 'PhonemeEmbed')

        # Stage 2: P2W
        with c.subgraph(name='cluster_p2w') as p2w:
            p2w.attr(label='Stage 2: Phoneme-to-Word', color='#f50057', fontcolor='#f50057', style='dotted')
            p2w.node('ReaderP2W', 'TinyReader\n(Sequential Integrator Bi-LSTM)', fillcolor='#c2185b')
            p2w.node('WordEmbed', 'Word Embedding\n(Many-to-One)', fillcolor='#00acc1', shape='parallelogram')
            
            p2w.edge('ReaderP2W', 'WordEmbed')
            
        c.edge('PhonemeEmbed', 'ReaderP2W')

    # Conectar Visual a Reader
    dot.edge('VisLogits', 'ReaderG2P', style='dashed', label='Input')

    # ==========================================
    # 3. PHONOLOGICAL PATHWAY (Dual Listeners)
    # ==========================================
    with dot.subgraph(name='cluster_audio') as c:
        c.attr(label='Phonological Pathway (The "Ears")', color='#00f2fe', fontcolor='#00f2fe', style='dashed', penwidth='2')
        
        # Phoneme Listener (Judge 1)
        with c.subgraph(name='cluster_phoneme_listener') as pl:
            pl.attr(label='Phoneme Listener (Judge 1)', color='#ba68c8', fontcolor='#ba68c8', style='dotted')
            pl.node('PL_Class', 'Phoneme Classifier', fillcolor='#7b1fa2')
            pl.node('PL_Pred', 'Predicted Phonemes', fillcolor='#8e24aa', shape='ellipse')
            pl.edge('PL_Class', 'PL_Pred')

        # Word Listener (Judge 2 & Target)
        with c.subgraph(name='cluster_word_listener') as wl:
            wl.attr(label='Word Listener (Judge 2)', color='#fdd835', fontcolor='#fdd835', style='dotted')
            
            # Target Generation
            wl.node('AudioInput', 'Real Audio\n(Spoken Word)', fillcolor='#1e88e5', shape='note')
            wl.node('AudioEnc', 'Audio Encoder\n(CNN + Transformer)', fillcolor='#006064')
            wl.node('RealEmbed', 'Real Embeddings\n(Target)', fillcolor='#00acc1', shape='parallelogram')
            
            wl.edge('AudioInput', 'AudioEnc')
            wl.edge('AudioEnc', 'RealEmbed')
            
            # Classification
            wl.node('WL_Class', 'Word Classifier', fillcolor='#f9a825')
            wl.node('WL_Pred', 'Predicted Word', fillcolor='#fbc02d', shape='ellipse')
            wl.edge('WL_Class', 'WL_Pred')

    # ==========================================
    # 4. LOSSES (Dual Objectives)
    # ==========================================
    with dot.subgraph(name='cluster_losses') as c:
        c.attr(label='Training Objectives', color='#fdd835', fontcolor='#fdd835', style='dotted', penwidth='2')
        
        c.node('G2P_Loss', 'G2P Losses\n(MSE + Cos + Perceptual)', fillcolor='#e53935', shape='diamond')
        c.node('P2W_Loss', 'P2W Losses\n(MSE + Cos + Perceptual)', fillcolor='#d32f2f', shape='diamond')

    # Conexiones de Loss
    # Stage 1 Loss (Mixed)
    dot.edge('PhonemeEmbed', 'PL_Class', label='Imagined Phonemes')
    dot.edge('PL_Pred', 'G2P_Loss')
    dot.edge('PhonemeEmbed', 'G2P_Loss', label='MSE/Cos vs Bank')
    
    # Stage 2 Loss (Mixed)
    dot.edge('WordEmbed', 'P2W_Loss', label='Imagined Word')
    dot.edge('RealEmbed', 'P2W_Loss', label='Ground Truth')
    dot.edge('WordEmbed', 'WL_Class')
    dot.edge('WL_Pred', 'P2W_Loss')

    return dot

def get_tiny_reader_diagram():
    dot = graphviz.Digraph('TinyReader', comment='TinyReader Two-Stage Architecture')
    dot.attr(rankdir='LR', bgcolor='#0e1117', fontname='Helvetica', fontcolor='white')
    
    # Global Styles
    dot.attr('node', **{'fontname': 'Helvetica', 'shape': 'box', 'style': 'filled,rounded', 'fontcolor': 'white', 'color': '#303030', 'margin': '0.2'})
    dot.attr('edge', **{'fontname': 'Helvetica', 'color': '#6c757d', 'fontcolor': '#adb5bd', 'penwidth': '1.5'})

    # --- STAGE 1: G2P ---
    with dot.subgraph(name='cluster_stage1') as s1:
        s1.attr(label='Stage 1: Grapheme-to-Phoneme (G2P)', color='#00f2fe', fontcolor='#00f2fe', style='dashed', penwidth='2')
        
        s1.node('InputText', 'Input Text\n"gato"', fillcolor='#1e88e5', shape='note')
        s1.node('VisualPathway', 'Visual Pathway\n(Frozen)', fillcolor='#424242', fontcolor='#bdbdbd')
        s1.node('SpellingEmb', 'Spelling Embeddings\n(Letters)', fillcolor='#00838f', shape='ellipse')
        
        # Reader G2P Internals (Pointwise Projector)
        with s1.subgraph(name='cluster_g2p_internal') as g2p_int:
            g2p_int.attr(label='TinySpeller (Pointwise MLP)', color='#ff4081', fontcolor='#ff4081', style='solid')
            g2p_int.node('G2P_Proj', 'Projector\n(Linear + GELU)', fillcolor='#880e4f')
            g2p_int.edge('SpellingEmb', 'G2P_Proj')
        
        s1.node('PhonemeEmb', 'Phoneme Embeddings\n(Predicted)', fillcolor='#fdd835', fontcolor='black', shape='ellipse')
        
        s1.node('PhonemeListener', 'Phoneme Listener\n(Frozen Judge)', fillcolor='#424242', fontcolor='#bdbdbd')
        s1.node('LossG2P', 'Loss G2P\n(Soft-DTW + Perceptual)', fillcolor='#e53935', shape='diamond')

        s1.edge('InputText', 'VisualPathway')
        s1.edge('VisualPathway', 'SpellingEmb')
        s1.edge('SpellingEmb', 'G2P_Enc')
        s1.edge('G2P_Proj', 'PhonemeEmb')
        s1.edge('PhonemeEmb', 'PhonemeListener')
        s1.edge('PhonemeListener', 'LossG2P')

    # --- STAGE 2: P2W ---
    with dot.subgraph(name='cluster_stage2') as s2:
        s2.attr(label='Stage 2: Phoneme-to-Word (P2W)', color='#ff0080', fontcolor='#ff0080', style='dashed', penwidth='2')
        
        # Input to Stage 2 is output of Stage 1 (Frozen)
        s2.node('ReaderG2P_Frozen', 'Reader G2P\n(Frozen)', fillcolor='#424242', fontcolor='#bdbdbd')
        s2.node('PhonemeEmb_In', 'Phoneme Embeddings\n(Input)', fillcolor='#fdd835', fontcolor='black', shape='ellipse')
        
        # Reader P2W Internals (Sequential Integrator)
        with s2.subgraph(name='cluster_p2w_internal') as p2w_int:
            p2w_int.attr(label='TinyReader (Bi-LSTM Integrator)', color='#f50057', fontcolor='#f50057', style='solid')
            p2w_int.node('P2W_Enc', 'Encoder\n(Bi-LSTM)', fillcolor='#880e4f')
            p2w_int.node('P2W_Collapse', 'Aggregation\n(Last Hidden)', fillcolor='#ad1457', shape='circle')
            p2w_int.node('P2W_Head', 'Projection Head\n(Linear)', fillcolor='#c2185b')
            
            p2w_int.edge('P2W_Enc', 'P2W_Collapse')
            p2w_int.edge('P2W_Collapse', 'P2W_Head')
        
        s2.node('WordEmb', 'Word Embeddings\n(Predicted)', fillcolor='#43a047', shape='ellipse')
        
        s2.node('WordListener', 'Word Listener\n(Frozen Judge)', fillcolor='#424242', fontcolor='#bdbdbd')
        s2.node('LossP2W', 'Loss P2W\n(Soft-DTW + Perceptual)', fillcolor='#e53935', shape='diamond')

        s2.edge('ReaderG2P_Frozen', 'PhonemeEmb_In')
        s2.edge('PhonemeEmb_In', 'P2W_Enc')
        s2.edge('P2W_Head', 'WordEmb')
        s2.edge('WordEmb', 'WordListener')
        s2.edge('WordListener', 'LossP2W')

    # Connect Stages conceptually
    dot.edge('PhonemeEmb', 'ReaderG2P_Frozen', style='dotted', label='Weights Transfer')

    return dot

def get_full_flow_diagram():
    """
    Diagrama completo del flujo de información y entrenamiento basado en el esquema manual.
    Refleja:
    1. Ruta Visual: Grafema -> TinyEyes -> Logits -> TinySpeller -> Embeddings Fonema
    2. Ruta Auditiva (Fonema): Audio -> TinyEars(Phoneme) -> Embeddings Fonema
    3. Ruta Auditiva (Palabra): Audio -> TinyEars(Words) -> Embeddings Palabra
    4. TinyReader: Embeddings Fonema -> Embeddings Palabra
    5. Losses: Cross Entropy (Perceptual), Soft-DTW (Alineación)
    """
    dot = graphviz.Digraph('FullFlow', comment='TinySpeak Full Flow')
    dot.attr(rankdir='LR', bgcolor='#0e1117', fontname='Helvetica', fontcolor='white')
    
    # Global Styles
    dot.attr('node', **{'fontname': 'Helvetica', 'shape': 'box', 'style': 'filled,rounded', 'fontcolor': 'white', 'color': '#303030', 'margin': '0.2'})
    dot.attr('edge', **{'fontname': 'Helvetica', 'color': '#6c757d', 'fontcolor': '#adb5bd', 'penwidth': '1.5'})

    # ==========================================
    # 1. VISUAL INPUT & TINY EYES
    # ==========================================
    with dot.subgraph(name='cluster_visual') as c:
        c.attr(style='invis')
        c.node('Grafema', 'Grafema\n(Imagen)', fillcolor='#1e88e5', shape='note')
        c.node('TinyEyes', 'TinyEyes\n(Visual Encoder)', fillcolor='#263238')
        c.node('Logits', 'Logits\n(Grafema)', fillcolor='#d81b60', shape='parallelogram')
        
        c.edge('Grafema', 'TinyEyes')
        c.edge('TinyEyes', 'Logits')

    # ==========================================
    # 2. TINY SPELLER (G2P)
    # ==========================================
    with dot.subgraph(name='cluster_speller') as c:
        c.attr(style='invis')
        c.node('TinySpeller', 'TinySpeller\n(G2P)', fillcolor='#880e4f')
        c.node('EmbFonema_Img', 'Imagen auditiva del fonema\n(Embeddings)', fillcolor='#fdd835', fontcolor='black', shape='parallelogram')
        
        c.edge('Logits', 'TinySpeller')
        c.edge('TinySpeller', 'EmbFonema_Img')

    # ==========================================
    # 3. TINY READER (P2W)
    # ==========================================
    with dot.subgraph(name='cluster_reader') as c:
        c.attr(style='invis')
        c.node('TinyReader', 'TinyReader\n(P2W)', fillcolor='#c2185b')
        c.node('EmbPalabra_Img', 'Imagen auditiva de la palabra\n(Embeddings)', fillcolor='#00acc1', shape='parallelogram')
        c.node('ClasificadorPalabra', 'Clasificador\nde palabras', fillcolor='#f9a825', shape='box')
        
        c.edge('EmbFonema_Img', 'TinyReader')
        c.edge('TinyReader', 'EmbPalabra_Img')
        c.edge('EmbPalabra_Img', 'ClasificadorPalabra')

    # ==========================================
    # 4. AUDIO PHONEME PATH (GROUND TRUTH 1)
    # ==========================================
    with dot.subgraph(name='cluster_audio_phoneme') as c:
        c.attr(style='invis')
        c.node('AudioFonema', 'Audio\n(Fonema)', fillcolor='#1e88e5', shape='note')
        c.node('TinyEarsPhoneme', 'TinyEars\n(Phoneme)', fillcolor='#7b1fa2')
        c.node('EmbFonema_Real', 'Embeddings\n(Fonemas)', fillcolor='#ab47bc', shape='parallelogram')
        c.node('ClasificadorFonema', 'Clasificador\n(Fonemas)', fillcolor='#8e24aa')
        
        c.edge('AudioFonema', 'TinyEarsPhoneme')
        c.edge('TinyEarsPhoneme', 'EmbFonema_Real')
        c.edge('EmbFonema_Real', 'ClasificadorFonema')

    # ==========================================
    # 5. AUDIO WORD PATH (GROUND TRUTH 2)
    # ==========================================
    with dot.subgraph(name='cluster_audio_word') as c:
        c.attr(style='invis')
        c.node('AudioPalabra', 'Audio\n(Palabra)', fillcolor='#1e88e5', shape='note')
        c.node('TinyEarsWords', 'TinyEars\n(Words)', fillcolor='#006064')
        c.node('EmbPalabra_Real', 'Embeddings\n(Palabra)', fillcolor='#00acc1', shape='parallelogram')
        c.node('ClasificadorPalabraReal', 'Clasificador\n(Palabras)', fillcolor='#00838f')
        
        c.edge('AudioPalabra', 'TinyEarsWords')
        c.edge('TinyEarsWords', 'EmbPalabra_Real')
        c.edge('EmbPalabra_Real', 'ClasificadorPalabraReal')

    # ==========================================
    # 6. LOSSES (RED ARROWS)
    # ==========================================
    
    # TinyEyes Loss
    dot.edge('Logits', 'TinyEyes', color='#ff5252', fontcolor='#ff5252', label='Cross Entropy Loss\n(grafema -> etiqueta)', dir='back')

    # TinySpeller Alignment & Perceptual Loss
    dot.edge('EmbFonema_Img', 'EmbFonema_Real', color='#ff5252', fontcolor='#ff5252', label='MSE (Direct Match)', style='dashed', dir='both')
    dot.edge('EmbFonema_Img', 'ClasificadorFonema', color='#ff5252', fontcolor='#ff5252', label='Cross Entropy Loss\n(Perceptual Loss)')

    # TinyReader Alignment & Perceptual Loss
    dot.edge('EmbPalabra_Img', 'EmbPalabra_Real', color='#ff5252', fontcolor='#ff5252', label='MSE (Léxico Match)', style='dashed', dir='both')
    dot.edge('EmbPalabra_Img', 'ClasificadorPalabraReal', color='#ff5252', fontcolor='#ff5252', label='Cross Entropy Loss\n(Perceptual Loss)')

    # TinyEars Internal Losses
    dot.edge('ClasificadorFonema', 'TinyEarsPhoneme', color='#ff5252', fontcolor='#ff5252', label='Cross Entropy Loss', dir='back')
    dot.edge('ClasificadorPalabraReal', 'TinyEarsWords', color='#ff5252', fontcolor='#ff5252', label='Cross Entropy Loss', dir='back')

    # Layout adjustments (invisible edges to force structure)
    dot.edge('TinyEyes', 'TinyEarsPhoneme', style='invis')
    dot.edge('TinyEarsPhoneme', 'TinyEarsWords', style='invis')

    return dot
