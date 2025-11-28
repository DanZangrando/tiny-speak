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

def get_reader_diagram():
    dot = graphviz.Digraph('TinyReader', comment='TinyReader Architecture')
    dot.attr(rankdir='TB', bgcolor='#0e1117', fontname='Helvetica', fontcolor='white')
    
    # Global Styles
    dot.attr('node', **{'fontname': 'Helvetica', 'shape': 'box', 'style': 'filled,rounded', 'fontcolor': 'white', 'color': '#303030', 'margin': '0.2'})
    dot.attr('edge', **{'fontname': 'Helvetica', 'color': '#6c757d', 'fontcolor': '#adb5bd', 'penwidth': '1.5'})

    # ==========================================
    # 1. VISUAL PATHWAY (Contexto)
    # ==========================================
    with dot.subgraph(name='cluster_visual') as c:
        c.attr(label='Visual Pathway (The "Eye")', color='#4facfe', fontcolor='#4facfe', style='dashed', penwidth='2')
        
        c.node('VisInput', 'Input Image\n(Grapheme)', fillcolor='#1e88e5', shape='note')
        c.node('VisCNN', 'Visual CNN\n(V1 -> IT)', fillcolor='#263238')
        c.node('VisLogits', 'Visual Logits\n(Concept Vector)', fillcolor='#d81b60', shape='parallelogram')
        
        c.edge('VisInput', 'VisCNN')
        c.edge('VisCNN', 'VisLogits')

    # ==========================================
    # 2. TINY READER (Generador)
    # ==========================================
    with dot.subgraph(name='cluster_reader') as c:
        c.attr(label='TinyReader (The "Inner Voice")', color='#ff0080', fontcolor='#ff0080', style='dashed', penwidth='2')
        
        c.node('ReaderInput', 'Input Concept\n(From Visual Pathway)', fillcolor='#d81b60', shape='parallelogram')
        c.node('ReaderLSTM', 'LSTM Decoder\n(Generates Sequence)', fillcolor='#880e4f')
        c.node('ReaderProj', 'Projection\n(Hidden -> 256)', fillcolor='#880e4f')
        c.node('GenEmbed', 'Imagined Embeddings\n(Batch, Time, 256)', fillcolor='#00acc1', shape='parallelogram')
        
        c.edge('ReaderInput', 'ReaderLSTM')
        c.edge('ReaderLSTM', 'ReaderProj')
        c.edge('ReaderProj', 'GenEmbed')

    # Conectar Visual a Reader
    dot.edge('VisLogits', 'ReaderInput', style='dashed', label='Input')

    # ==========================================
    # 3. PHONOLOGICAL PATHWAY (Contexto / Target)
    # ==========================================
    with dot.subgraph(name='cluster_audio') as c:
        c.attr(label='Phonological Pathway (The "Ear")', color='#00f2fe', fontcolor='#00f2fe', style='dashed', penwidth='2')
        
        # Target Generation (Bottom-Up)
        c.node('AudioInput', 'Real Audio\n(Spoken Word)', fillcolor='#1e88e5', shape='note')
        c.node('MelSpec', 'Mel Spectrogram\n(Cochlea)', fillcolor='#0097a7')
        c.node('AudioCNN', 'Audio CNN\n(Feature Extractor)', fillcolor='#00838f')
        c.node('AudioTrans', 'Transformer\n(Context Encoder)', fillcolor='#006064')
        c.node('RealEmbed', 'Real Embeddings\n(Target)', fillcolor='#00acc1', shape='parallelogram')
        
        c.edge('AudioInput', 'MelSpec')
        c.edge('MelSpec', 'AudioCNN')
        c.edge('AudioCNN', 'AudioTrans')
        c.edge('AudioTrans', 'RealEmbed')
        
        # Inner Ear (Classifier for Perceptual Loss)
        with c.subgraph(name='cluster_innerear') as ie:
            ie.attr(label='Inner Ear (Classifier)', color='#fdd835', fontcolor='#fdd835', style='dotted')
            ie.node('ListenerClass', 'Classifier Head\n(MeanPool + Linear)', fillcolor='#f9a825')
            ie.node('ListenerPred', 'Predicted Word\n(Understanding)', fillcolor='#fbc02d', shape='ellipse')
            
            ie.edge('ListenerClass', 'ListenerPred')

    # ==========================================
    # 4. LOSSES (Comparaciones)
    # ==========================================
    with dot.subgraph(name='cluster_losses') as c:
        c.attr(label='Training Objectives', color='#fdd835', fontcolor='#fdd835', style='dotted', penwidth='2')
        
        c.node('MSE', 'MSE Loss\n(Reconstruction)', fillcolor='#e53935', shape='diamond')
        c.node('Cos', 'Cosine Loss\n(Structure)', fillcolor='#e53935', shape='diamond')
        c.node('Perceptual', 'Perceptual Loss\n(Intelligibility)', fillcolor='#e53935', shape='diamond')

    # Conexiones de Loss
    dot.edge('GenEmbed', 'MSE')
    dot.edge('RealEmbed', 'MSE')
    
    dot.edge('GenEmbed', 'Cos')
    dot.edge('RealEmbed', 'Cos')
    
    # Flujo Perceptual
    dot.edge('GenEmbed', 'ListenerClass', label='Imagined Input')
    dot.edge('ListenerPred', 'Perceptual', label='Prediction')
    dot.edge('VisInput', 'Perceptual', label='Ground Truth Label', style='dotted') # El label viene de la letra/palabra

    return dot
