/**
 * @file LayerType.h
 * @author Robin MENEUST
 * @brief Enumeration of the type of layers in an AI model
 * @date 2024-01-13
 */

#ifndef CPP_AI_PROJECT_LAYER_TYPE_H
#define CPP_AI_PROJECT_LAYER_TYPE_H

/**
 * Type of layers that can be added to the AI model
 */
enum class LayerType {
    Dense, /*!< Dense layer  */
    Flatten, /*!< Flatten layer */
};

#endif
