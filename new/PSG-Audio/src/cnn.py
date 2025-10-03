def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(32, kernel_size=11, activation='relu', input_shape=input_shape, padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=3),
        Dropout(0.2),
        
        Conv1D(64, kernel_size=7, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=3),
        Dropout(0.3),
        
        Conv1D(128, kernel_size=5, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        GlobalAveragePooling1D(),
        
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(num_classes, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def train_cnn(X_train, y_train, X_val, y_val, class_weights=None):
    model = create_cnn_model(X_train.shape[1:], y_train.shape[1])
    
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7, verbose=1)
    ]
    
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    return model, history

def evaluate_cnn(model, X_test, y_test):
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    return acc

def run_cnn_on_features(X_train, y_train, X_test, y_test, class_weights=None):
    model, history = train_cnn(X_train, y_train, X_test, y_test, class_weights)
    acc = evaluate_cnn(model, X_test, y_test)
    return acc, history