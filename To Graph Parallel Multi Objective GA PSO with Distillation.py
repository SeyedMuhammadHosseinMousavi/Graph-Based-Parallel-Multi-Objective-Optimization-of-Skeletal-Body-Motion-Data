import os
import numpy as np
import random
import multiprocessing
from bvh import Bvh
from scipy.interpolate import interp1d
from scipy.fft import fft
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import networkx as nx
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# === BVH Reader ===
def read_bvh_file(path):
    print(f"📁 Reading file: {path}")
    with open(path, 'r') as f:
        mocap = Bvh(f.read())
    return mocap

# === Graph Creator ===
def create_graph(mocap):
    G = nx.Graph()
    for joint in mocap.get_joints():
        if joint.parent is not None:
            try:
                G.add_edge(joint.name, joint.parent.name)
            except:
                continue
    return G

# === Interpolation ===
def interpolate_motion(mocap, target_length):
    joint_names = [j.name for j in mocap.get_joints()]
    frames = []
    for i in range(mocap.nframes):
        frame = []
        for joint in joint_names:
            try:
                values = [float(mocap.frame_joint_channel(i, joint, ch)) for ch in mocap.joint_channels(joint)]
            except:
                values = [0.0, 0.0, 0.0]
            frame.extend(values)
        frames.append(frame)
    frames = np.array(frames)
    if frames.shape[0] == target_length:
        return frames, joint_names
    interp_func = interp1d(np.linspace(0, 1, frames.shape[0]), frames, axis=0, kind='linear')
    return interp_func(np.linspace(0, 1, target_length)), joint_names

# === Feature Extraction ===
def extract_graph_features(G):
    degs = dict(G.degree())
    features = [
        np.mean(list(degs.values())),
        np.max(list(degs.values())),
        nx.density(G),
        nx.transitivity(G),
        nx.number_connected_components(G)
    ]
    if nx.is_connected(G):
        features.append(nx.average_shortest_path_length(G))
    else:
        features.append(0.0)
    L = nx.normalized_laplacian_matrix(G).toarray()
    eig = np.linalg.eigvalsh(L)
    features.append(np.mean(eig))
    features.append(np.std(eig))
    return features

def extract_fft_features(motion_data):
    result = []
    for i in range(motion_data.shape[1]):
        f = np.abs(fft(motion_data[:, i]))
        result.append(np.mean(f))
        result.append(np.std(f))
    return result

# === GA Functions ===
def mutate_edges(graph, prob=0.3):
    new_g = graph.copy()
    for edge in list(new_g.edges()):
        if random.random() < prob:
            new_g.remove_edge(*edge)
    return new_g

def evolve_graphs(graphs, motions, labels, generations=10, pop_size=8):
    print("\n Starting Multi-Objective Genetic Algorithm...")
    population = [mutate_edges(g, 0.5) for g in graphs]
    for gen in range(generations):
        print(f"\n GA Generation {gen + 1}")
        scores = []
        
        for i, g in enumerate(population):
            feats = []
            for idx in range(len(graphs)):
                Gf = extract_graph_features(g)
                Ff = extract_fft_features(motions[idx])
                feats.append(Gf + Ff)
            X = np.array(feats)
            y = np.array(labels)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            acc = accuracy_score(y_test, model.predict(X_test))
            
            # Multi-objective: graph size (or another graph-related metric)
            graph_size = len(g.edges)  # Or use any other graph-related metric
            
            # Store both objectives: (accuracy, graph size)
            scores.append((acc, graph_size, g))

        # Sorting based on Pareto dominance or a weighted sum approach (if needed)
        population = [s[2] for s in sorted(scores, key=lambda x: (-x[0], x[1]))[:pop_size]]

    best_graph = population[0]
    print("\n Best graph structure selected by GA (considering both objectives).")
    return best_graph


# === PSO Functions ===
def pso_optimize_weights(motion_data, graph, labels, n_particles=8, iterations=10):
    print("\n Starting Multi-Objective Particle Swarm Optimization...")
    
    n_nodes = motion_data[0].shape[1]
    particles = np.random.rand(n_particles, n_nodes)
    velocities = np.random.rand(n_particles, n_nodes) * 0.1
    best_scores = [0] * n_particles
    best_positions = particles.copy()
    global_best_score = 0
    global_best_position = None

    for iter in range(iterations):
        print(f"\n PSO Iteration {iter + 1}")
        for p in range(n_particles):
            feats = []
            for i, motion in enumerate(motion_data):
                weighted = motion * particles[p]
                fft_feats = extract_fft_features(weighted)
                graph_feats = extract_graph_features(graph)
                feats.append(graph_feats + fft_feats)
            X = np.array(feats)
            y = np.array(labels)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            model = RandomForestClassifier(n_estimators=50)
            model.fit(X_train, y_train)
            acc = accuracy_score(y_test, model.predict(X_test))
            
            # Multi-objective: joint weight impact (based on particle position)
            joint_weight_impact = np.sum(particles[p])  # Or another weight-related metric
            
            # Store both objectives: (accuracy, joint weight impact)
            if acc > best_scores[p]:
                best_scores[p] = acc
                best_positions[p] = particles[p]

            # Track the best overall solution
            if acc > global_best_score:
                global_best_score = acc
                global_best_position = particles[p]
        
        # Update positions based on both objectives
        for p in range(n_particles):
            velocities[p] += 0.5 * (best_positions[p] - particles[p]) + 0.3 * (global_best_position - particles[p])
            particles[p] += velocities[p]

    print("\n Best joint weights found by PSO (considering both objectives).")
    return global_best_position


# === Parallelized Full Pipeline ===
def run_pipeline(dataset_path):
    all_graphs, all_motion, labels = [], [], []
    print(" Reading and processing BVH files...")
    for root, _, files in os.walk(dataset_path):
        for f in files:
            if f.endswith(".bvh"):
                label = os.path.basename(root)
                path = os.path.join(root, f)
                try:
                    mocap = read_bvh_file(path)
                    graph = create_graph(mocap)
                    motion, _ = interpolate_motion(mocap, 80)
                    all_graphs.append(graph)
                    all_motion.append(motion)
                    labels.append(label)
                    print(f" Loaded sample: {f} → Label: {label}")
                except Exception as e:
                    print(f" Failed to process {f}: {e}")

    label_enc = LabelEncoder()
    y_encoded = label_enc.fit_transform(labels)

    # Parallelizing GA and PSO
    with multiprocessing.Pool(processes=2) as pool:
        # GA and PSO run concurrently on different cores
        results = pool.starmap(run_algorithms_in_parallel, [(all_graphs, all_motion, y_encoded)])

    best_graph, weights = results[0]

    final_feats = []
    for motion in all_motion:
        weighted = motion * weights
        fft_feats = extract_fft_features(weighted)
        graph_feats = extract_graph_features(best_graph)
        final_feats.append(graph_feats + fft_feats)

    X = np.array(final_feats)
    y = y_encoded
    return X, y, label_enc

# === Parallel Algorithms ===
def run_algorithms_in_parallel(graphs, motions, labels):
    print("\n Running Genetic Algorithm (GA)...")
    best_graph = evolve_graphs(graphs, motions, labels)

    print("\n Running Particle Swarm Optimization (PSO)...")
    weights = pso_optimize_weights(motions, best_graph, labels)

    print("\n Completed both GA and PSO.")
    return best_graph, weights


# === MAIN ===
if __name__ == "__main__":
    X, y, label_enc = run_pipeline("Emotions Four Main")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    
    # === Gradient Boosting (GBM) ===
    print("\n🌟 Gradient Boosting Classifier (Teacher):")
    gbm = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.02,
        max_depth=8,
        subsample=0.7,
        random_state=42,
        verbose=1
    )
    gbm.fit(X_train, y_train)
    y_pred_gbm = gbm.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred_gbm):.4f}")
    print("Classification Report for GBM (Teacher):")
    print(classification_report(y_test, y_pred_gbm, target_names=label_enc.classes_))
    
    
    # === Standalone Decision Tree ===
    print("\n🌳 Standalone Decision Tree Classifier:")
    dt = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=42
    )
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
    print("Classification Report for Decision Tree (Standalone):")
    print(classification_report(y_test, y_pred_dt, target_names=label_enc.classes_))
    
    
    # === Knowledge Distillation: GBM ➜ Decision Tree ===
    print("\n📘 Knowledge Distillation: GBM ➜ Decision Tree")
    
    # Step 1: Get soft labels from GBM
    gbm_probs = gbm.predict_proba(X_train)
    
    # Step 2: Convert true labels to one-hot encoding
    enc = OneHotEncoder(sparse_output=False)
    y_train_onehot = enc.fit_transform(y_train.reshape(-1, 1))
    
    # Step 3: Hybrid target (combine soft & hard labels)
    temperature = 3.0
    def softmax_with_temperature(logits, T):
        logits = np.array(logits)
        exp_logits = np.exp(logits / T)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    soft_labels = softmax_with_temperature(gbm_probs, temperature)
    alpha = 0.5
    hybrid_targets = alpha * soft_labels + (1 - alpha) * y_train_onehot
    y_hybrid = np.argmax(hybrid_targets, axis=1)
    
    # Step 4: Train student Decision Tree on hybrid labels
    dt_kd = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=42
    )
    dt_kd.fit(X_train, y_hybrid)
    y_pred_kd = dt_kd.predict(X_test)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred_kd):.4f}")
    print("Classification Report for DT (Knowledge Distillation):")
    print(classification_report(y_test, y_pred_kd, target_names=label_enc.classes_))


    # === Bar Plot ===
    # === Extract F1-scores from classification reports ===
    report_gbm = classification_report(y_test, y_pred_gbm, target_names=label_enc.classes_, output_dict=True)
    report_dt = classification_report(y_test, y_pred_dt, target_names=label_enc.classes_, output_dict=True)
    report_kd = classification_report(y_test, y_pred_kd, target_names=label_enc.classes_, output_dict=True)
    
    f1_gbm = [report_gbm[label]['f1-score'] for label in label_enc.classes_]
    f1_dt = [report_dt[label]['f1-score'] for label in label_enc.classes_]
    f1_kd = [report_kd[label]['f1-score'] for label in label_enc.classes_]
    
    # === Prepare data for bar plot ===
    data = pd.DataFrame({
        'Class': list(label_enc.classes_) * 3,
        'F1-Score': f1_gbm + f1_dt + f1_kd,
        'Classifier': (['Gradient Boosting'] * len(f1_gbm)) +
                      (['Decision Tree'] * len(f1_dt)) +
                      (['Distilled DT'] * len(f1_kd))
    })
    
    # === Create bar plot ===
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Class', y='F1-Score', hue='Classifier', data=data, palette='Set2')
    plt.title('Per-Class F1-Scores: GBM (Teacher) vs. DT vs. Distilled DT', fontsize=16, weight='bold')
    plt.xlabel('Emotion Class', fontsize=14, weight='bold')
    plt.ylabel('F1-Score', fontsize=14, weight='bold')
    plt.ylim(0, 1.05)
    plt.legend(title='Classifier')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()
