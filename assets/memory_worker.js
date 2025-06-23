// Memory Worker for Synaptic WebAssembly
// Handles heavy memory operations in background thread

// In-memory storage for the worker
let memoryStore = new Map();
let compressionEnabled = true;

// Simple compression using browser's built-in compression
async function compressData(data) {
    if (!compressionEnabled) return data;
    
    try {
        const stream = new CompressionStream('gzip');
        const writer = stream.writable.getWriter();
        const reader = stream.readable.getReader();
        
        writer.write(data);
        writer.close();
        
        const chunks = [];
        let done = false;
        
        while (!done) {
            const { value, done: readerDone } = await reader.read();
            done = readerDone;
            if (value) chunks.push(value);
        }
        
        return new Uint8Array(chunks.reduce((acc, chunk) => [...acc, ...chunk], []));
    } catch (error) {
        console.warn('Compression failed, using uncompressed data:', error);
        return data;
    }
}

// Simple decompression using browser's built-in decompression
async function decompressData(data) {
    if (!compressionEnabled) return data;
    
    try {
        const stream = new DecompressionStream('gzip');
        const writer = stream.writable.getWriter();
        const reader = stream.readable.getReader();
        
        writer.write(data);
        writer.close();
        
        const chunks = [];
        let done = false;
        
        while (!done) {
            const { value, done: readerDone } = await reader.read();
            done = readerDone;
            if (value) chunks.push(value);
        }
        
        return new Uint8Array(chunks.reduce((acc, chunk) => [...acc, ...chunk], []));
    } catch (error) {
        console.warn('Decompression failed, returning original data:', error);
        return data;
    }
}

// Store data in memory
async function storeData(key, data, compress = true) {
    try {
        const dataToStore = compress ? await compressData(new Uint8Array(data)) : new Uint8Array(data);
        memoryStore.set(key, {
            data: dataToStore,
            compressed: compress,
            timestamp: Date.now()
        });
        return { success: true };
    } catch (error) {
        return { success: false, error: error.message };
    }
}

// Retrieve data from memory
async function retrieveData(key) {
    try {
        const entry = memoryStore.get(key);
        if (!entry) {
            return { success: true, data: null };
        }
        
        const data = entry.compressed ? await decompressData(entry.data) : entry.data;
        return { success: true, data: Array.from(data) };
    } catch (error) {
        return { success: false, error: error.message };
    }
}

// Delete data from memory
function deleteData(key) {
    try {
        const existed = memoryStore.has(key);
        memoryStore.delete(key);
        return { success: true, existed };
    } catch (error) {
        return { success: false, error: error.message };
    }
}

// List all keys
function listKeys() {
    try {
        const keys = Array.from(memoryStore.keys());
        return { success: true, keys };
    } catch (error) {
        return { success: false, error: error.message };
    }
}

// Simple text search across stored data
async function searchData(query, limit = 10) {
    try {
        const results = [];
        const queryLower = query.toLowerCase();
        
        for (const [key, entry] of memoryStore.entries()) {
            try {
                // Decompress data for searching
                const data = entry.compressed ? await decompressData(entry.data) : entry.data;
                const text = new TextDecoder().decode(data);
                const textLower = text.toLowerCase();
                
                if (textLower.includes(queryLower)) {
                    // Calculate simple relevance score
                    const occurrences = (textLower.match(new RegExp(queryLower, 'g')) || []).length;
                    const score = occurrences / text.length;
                    
                    // Create snippet
                    const index = textLower.indexOf(queryLower);
                    const start = Math.max(0, index - 50);
                    const end = Math.min(text.length, index + query.length + 50);
                    const snippet = text.substring(start, end);
                    
                    results.push({
                        key,
                        score,
                        snippet: start > 0 ? '...' + snippet : snippet
                    });
                }
            } catch (error) {
                // Skip entries that can't be decoded
                continue;
            }
        }
        
        // Sort by score and limit results
        results.sort((a, b) => b.score - a.score);
        return { success: true, results: results.slice(0, limit) };
    } catch (error) {
        return { success: false, error: error.message };
    }
}

// Message handler
self.onmessage = async function(event) {
    const { request_id, message } = event.data;
    
    try {
        let response;
        
        switch (message.type) {
            case 'Store':
                const storeResult = await storeData(message.key, message.data, message.compress);
                if (storeResult.success) {
                    response = {
                        type: 'Success',
                        request_id,
                        data: null
                    };
                } else {
                    response = {
                        type: 'Error',
                        request_id,
                        message: storeResult.error
                    };
                }
                break;
                
            case 'Retrieve':
                const retrieveResult = await retrieveData(message.key);
                if (retrieveResult.success) {
                    response = {
                        type: 'Success',
                        request_id,
                        data: retrieveResult.data
                    };
                } else {
                    response = {
                        type: 'Error',
                        request_id,
                        message: retrieveResult.error
                    };
                }
                break;
                
            case 'Delete':
                const deleteResult = deleteData(message.key);
                if (deleteResult.success) {
                    response = {
                        type: 'Success',
                        request_id,
                        data: null
                    };
                } else {
                    response = {
                        type: 'Error',
                        request_id,
                        message: deleteResult.error
                    };
                }
                break;
                
            case 'ListKeys':
                const listResult = listKeys();
                if (listResult.success) {
                    response = {
                        type: 'Keys',
                        request_id,
                        keys: listResult.keys
                    };
                } else {
                    response = {
                        type: 'Error',
                        request_id,
                        message: listResult.error
                    };
                }
                break;
                
            case 'Compress':
                const compressed = await compressData(new Uint8Array(message.data));
                response = {
                    type: 'Success',
                    request_id,
                    data: Array.from(compressed)
                };
                break;
                
            case 'Decompress':
                const decompressed = await decompressData(new Uint8Array(message.data));
                response = {
                    type: 'Success',
                    request_id,
                    data: Array.from(decompressed)
                };
                break;
                
            case 'Search':
                const searchResult = await searchData(message.query, message.limit);
                if (searchResult.success) {
                    response = {
                        type: 'SearchResults',
                        request_id,
                        results: searchResult.results
                    };
                } else {
                    response = {
                        type: 'Error',
                        request_id,
                        message: searchResult.error
                    };
                }
                break;
                
            default:
                response = {
                    type: 'Error',
                    request_id,
                    message: `Unknown message type: ${message.type}`
                };
        }
        
        self.postMessage(response);
    } catch (error) {
        self.postMessage({
            type: 'Error',
            request_id,
            message: error.message
        });
    }
};

// Worker initialization
console.log('Synaptic Memory Worker initialized');

// Check for compression support
if (typeof CompressionStream === 'undefined' || typeof DecompressionStream === 'undefined') {
    console.warn('Browser compression not supported, disabling compression');
    compressionEnabled = false;
}
