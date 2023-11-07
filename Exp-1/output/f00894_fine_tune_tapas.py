from typing import *
import tensorflow as tf
from transformers import TapasConfig, TFTapasForQuestionAnswering

def fine_tune_tapas(train_dataloader):
	config = TapasConfig(
		num_aggregation_labels=4,
		use_answer_as_supervision=True,
		answer_loss_cutoff=0.664694,
		cell_selection_preference=0.207951,
		huber_loss_delta=0.121194,
		init_cell_selection_weights_to_zero=True,
		select_one_column=True,
		allow_empty_column_selection=False,
		temperature=0.0352513,
	)
	model = TFTapasForQuestionAnswering.from_pretrained("google/tapas-base", config=config)

	optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

	for epoch in range(2):
		for batch in train_dataloader:
			input_ids = batch[0]
			attention_mask = batch[1]
			token_type_ids = batch[4]
			labels = batch[-1]
			numeric_values = batch[2]
			numeric_values_scale = batch[3]
			float_answer = batch[6]

			with tf.GradientTape() as tape:
				outputs = model(
					input_ids=input_ids,
					attention_mask=attention_mask,
					token_type_ids=token_type_ids,
					labels=labels,
					numeric_values=numeric_values,
					numeric_values_scale=numeric_values_scale,
					float_answer=float_answer,
				)
			grads = tape.gradient(outputs.loss, model.trainable_weights)
			optimizer.apply_gradients(zip(grads, model.trainable_weights))
