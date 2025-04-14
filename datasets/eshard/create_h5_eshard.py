import numpy as np
import h5py
import estraces

ths = estraces.read_ths_from_ets_file("./Nucleo_AES_masked_non_shuffled.ets")
n_profiling = 90000
n_attack = 10000

raw_plaintexts=ths.metadatas["plaintext"]
raw_keys = ths.metadatas["key"]
raw_masks = ths.metadatas["mask"]
profiling_samples = ths.samples[0:n_profiling]
profiling_plaintext = raw_plaintexts[0:n_profiling]
profiling_key = raw_keys[0:n_profiling]
profiling_masks = raw_masks[0:n_profiling]

attack_samples = ths.samples[n_profiling:n_profiling+n_attack]
attack_plaintext = raw_plaintexts[n_profiling:n_profiling + n_attack]
attack_key = raw_keys[n_profiling:n_profiling + n_attack]
attack_masks = raw_masks[n_profiling:n_profiling + n_attack]

out_file = h5py.File('./eshard.h5', 'w')

profiling_index = [n for n in range(n_profiling)]
attack_index = [n for n in range(n_attack)]

profiling_traces_group = out_file.create_group("Profiling_traces")
attack_traces_group = out_file.create_group("Attack_traces")

profiling_traces_group.create_dataset(name="traces", data=profiling_samples, dtype=profiling_samples.dtype)
attack_traces_group.create_dataset(name="traces", data=attack_samples, dtype=attack_samples.dtype)

metadata_type_profiling = np.dtype([("plaintext", profiling_plaintext.dtype, (len(profiling_plaintext[0]),)),
                                    ("key", profiling_key.dtype, (len(profiling_key[0]),)),
                                    ("masks", profiling_masks.dtype, (len(profiling_masks[0]),))
                                    ])
metadata_type_attack = np.dtype([("plaintext", attack_plaintext.dtype, (len(attack_plaintext[0]),)),
                                    ("key", attack_key.dtype, (len(attack_key[0]),)),
                                    ("masks", attack_masks.dtype, (len(attack_masks[0]),))
                                    ])

profiling_metadata = np.array([(profiling_plaintext[n], profiling_key[n], profiling_masks[n]) for n in
                                profiling_index], dtype=metadata_type_profiling)
profiling_traces_group.create_dataset("metadata", data=profiling_metadata, dtype=metadata_type_profiling)

attack_metadata = np.array([(attack_plaintext[n], attack_key[n], attack_masks[n]) for n in
                            attack_index], dtype=metadata_type_attack)
attack_traces_group.create_dataset("metadata", data=attack_metadata, dtype=metadata_type_attack)

out_file.flush()
out_file.close()
