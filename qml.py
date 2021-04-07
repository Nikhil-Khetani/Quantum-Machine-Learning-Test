from qiskit_machine_learning.datasets import ad_hoc_data
from qiskit_machine_learning.algorithms import VQC
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.algorithms.optimizers import L_BFGS_B
from qiskit.providers.aer import QasmSimulator
X_train, y_train, X_test, y_test = ad_hoc_data(20, 10, 2, 0.1)
num_qubits = 2
vqc = VQC(feature_map=ZZFeatureMap(num_qubits),
        ansatz = RealAmplitudes(num_qubits, reps=1),
        loss = 'cross_entropy',
        optimizer = L_BFGS_B(),
        quantum_instance = QasmSimulator())

vqc.fit(X_train,y_train)
vqc.score(X_test, y_test)